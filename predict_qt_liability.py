#!/usr/bin/env python3
"""
Apply the QT-M2M4 ensemble model to user-supplied compounds.

Purpose
-------
This script provides a simple inference entry point for research users who wish to inspect compound-level QT-M2M4 prioritization behavior.
It reads a CSV containing SMILES, constructs graph objects using the same feature
scheme used in the manuscript workflow, loads the trained M2 and M4 family models,
and writes ensemble predictions with the fixed manuscript thresholds.

Expected input CSV
------------------
A header row containing at least one SMILES column. Accepted column names include:
    - SMILES
    - smiles
    - SMILES_ISO

An optional identifier column may also be supplied. Accepted names include:
    - compound_id
    - Compound_ID
    - id
    - name
    - drug

Output columns
--------------
    compound_id
    smiles_input
    smiles_canonical
    valid_rdkit
    p_m2
    p_m4
    p_ens
    binary_call_055
    triage_label
    note

Notes
-----
- The final ensemble score is a weighted average of the M2 and M4 family scores.
- By default, family weights are inferred from validation PR-AUC summary files.
- The manuscript decision thresholds are used by default:
      binary threshold = 0.55
      GREEN boundary   = 0.44
      RED boundary     = 0.75
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdchem

try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import (
        AttentionalAggregation,
        GATv2Conv,
        NNConv,
        TransformerConv,
    )
except Exception as e:
    raise SystemExit(
        "torch-geometric is required for predict_qt_liability.py. "
        "Please install the packages listed in requirements.txt before running this script.\n"
        f"Original import error: {e}"
    )

RDLogger.DisableLog("rdApp.*")

SEED = 12345
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEFAULT_BINARY_THRESHOLD = 0.55
DEFAULT_GREEN_THRESHOLD = 0.44
DEFAULT_RED_THRESHOLD = 0.75
DEFAULT_BATCH_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply the QT-M2M4 ensemble model to user compounds")
    parser.add_argument("--input", required=True, help="Input CSV containing a SMILES column")
    parser.add_argument("--output", required=True, help="Output CSV for predictions")
    parser.add_argument(
        "--root",
        default=".",
        help="Repository root containing model folders, scaler files, and workflow artifacts (default: current directory)",
    )
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for inference")
    parser.add_argument("--binary_thr", type=float, default=DEFAULT_BINARY_THRESHOLD, help="Binary decision threshold")
    parser.add_argument("--thr_green", type=float, default=DEFAULT_GREEN_THRESHOLD, help="GREEN/YELLOW boundary")
    parser.add_argument("--thr_red", type=float, default=DEFAULT_RED_THRESHOLD, help="YELLOW/RED boundary")
    parser.add_argument(
        "--smiles_col",
        default=None,
        help="Explicit SMILES column name. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--id_col",
        default=None,
        help="Explicit identifier column name. If omitted, common names are auto-detected.",
    )
    parser.add_argument(
        "--family_weight_mode",
        choices=["auto", "equal"],
        default="auto",
        help="How to combine family scores. 'auto' uses validation PR-AUC summaries. 'equal' uses 0.5 / 0.5.",
    )
    return parser.parse_args()


def find_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    low = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        c = low.get(str(n).strip().lower())
        if c:
            return c
    return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if smiles is None:
        return None
    s = str(smiles).strip()
    if not s:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


# ---------- Descriptor / graph features (matching notebook workflow) ----------
def gfeat7(mol):
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.TPSA(mol),
        Descriptors.HeavyAtomCount(mol),
        Descriptors.RingCount(mol),
    ]


def rdesc10(mol):
    return [
        Descriptors.FpDensityMorgan1(mol),
        Descriptors.FpDensityMorgan2(mol),
        Descriptors.FpDensityMorgan3(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.BalabanJ(mol),
        Descriptors.BertzCT(mol),
        Descriptors.FractionCSP3(mol),
    ]


def atom_f(atom: rdchem.Atom) -> torch.Tensor:
    return torch.tensor(
        [
            atom.GetAtomicNum(),
            atom.GetTotalDegree(),
            atom.GetFormalCharge(),
            float(atom.GetIsAromatic()),
            atom.GetTotalNumHs(includeNeighbors=True),
        ],
        dtype=torch.float,
    )


def bond_f(bond: rdchem.Bond) -> torch.Tensor:
    return torch.tensor(
        [
            bond.GetBondTypeAsDouble(),
            float(bond.GetIsConjugated()),
            float(bond.IsInRing()),
            int(bond.GetStereo()),
            bond.GetBeginAtom().GetAtomicNum(),
            bond.GetEndAtom().GetAtomicNum(),
        ],
        dtype=torch.float,
    )


def mol_to_data(smiles: str, compound_id: str, scaler_g, scaler_r) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atoms = [atom_f(a) for a in mol.GetAtoms()]
    if not atoms:
        return None
    x = torch.stack(atoms)

    edge_pairs = []
    edge_attrs = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_f(bond)
        edge_pairs.extend([[i, j], [j, i]])
        edge_attrs.extend([bf, bf])

    if edge_pairs:
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        edge_attr = torch.stack(edge_attrs)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 6), dtype=torch.float)

    g = torch.tensor(scaler_g.transform([gfeat7(mol)]), dtype=torch.float)
    r = torch.tensor(scaler_r.transform([rdesc10(mol)]), dtype=torch.float)

    d = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, g=g, r=r)
    d.smiles_input = smiles
    d.smiles_canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    d.compound_id = str(compound_id)
    return d


# ---------- Models (adapted from strict external evaluation script) ----------
def make_edge_mlp(edge_dim: int, hid: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(edge_dim, hid), nn.ReLU(), nn.Linear(hid, out_dim * hid))


class Transformer_DX_DG_DR(nn.Module):
    def __init__(self, x_dim: int, g_dim: int, r_dim: int, e_dim: int, hid: int = 256):
        super().__init__()
        self.c1 = TransformerConv(x_dim, hid, heads=4, concat=False, edge_dim=e_dim, dropout=0.0, beta=False)
        self.c2 = TransformerConv(hid, hid, heads=4, concat=False, edge_dim=e_dim, dropout=0.0, beta=False)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hid, 1))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hid + g_dim + r_dim, 1)

    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index, d.edge_attr))
        x = F.relu(self.c2(x, d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim() == 3 else d.g
        r = d.r.squeeze(1) if d.r.dim() == 3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)


class NNConv_DX_DG_DR(nn.Module):
    def __init__(self, x_dim: int, g_dim: int, r_dim: int, e_dim: int, hid: int = 256):
        super().__init__()
        edge_nn1 = make_edge_mlp(e_dim, hid, x_dim)
        edge_nn2 = make_edge_mlp(e_dim, hid, hid)
        self.c1 = NNConv(x_dim, hid, edge_nn1, aggr="mean")
        self.c2 = NNConv(hid, hid, edge_nn2, aggr="mean")
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hid, 1))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hid + g_dim + r_dim, 1)

    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index, d.edge_attr))
        x = F.relu(self.c2(x, d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim() == 3 else d.g
        r = d.r.squeeze(1) if d.r.dim() == 3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)


class GATv2_DX_DG_DR(nn.Module):
    def __init__(self, x_dim: int, g_dim: int, r_dim: int, e_dim: int, hid: int = 256):
        super().__init__()
        self.g1 = GATv2Conv(x_dim, hid, edge_dim=e_dim, dropout=0.0)
        self.g2 = GATv2Conv(hid, hid, edge_dim=e_dim, dropout=0.0)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hid, 1))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hid + g_dim + r_dim, 1)

    def forward(self, d):
        x = F.elu(self.g1(d.x, d.edge_index, d.edge_attr))
        x = F.elu(self.g2(x, d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim() == 3 else d.g
        r = d.r.squeeze(1) if d.r.dim() == 3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)


class GATv2_DX_DG_DR_H4(nn.Module):
    def __init__(self, x_dim: int, g_dim: int, r_dim: int, e_dim: int, hid: int = 256):
        super().__init__()
        self.g1 = GATv2Conv(x_dim, hid, heads=4, concat=True, edge_dim=e_dim, dropout=0.0)
        self.g2 = GATv2Conv(hid * 4, hid, heads=4, concat=True, edge_dim=e_dim, dropout=0.0)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(hid * 4, 1))
        self.drop = nn.Dropout(0.2)
        self.fc = nn.Linear(hid * 4 + g_dim + r_dim, 1)

    def forward(self, d):
        x = F.elu(self.g1(d.x, d.edge_index, d.edge_attr))
        x = F.elu(self.g2(x, d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim() == 3 else d.g
        r = d.r.squeeze(1) if d.r.dim() == 3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)


# ---------- Weight discovery / loading ----------
def list_weight_files(folder: Path, patterns: Sequence[str]) -> List[str]:
    files: List[str] = []
    for pat in patterns:
        files += sorted(glob.glob(str(folder / pat)))
    return sorted(set(files))


def infer_weights_from_val_csv(csv_paths: Sequence[str]) -> Optional[float]:
    if not csv_paths:
        return None
    try:
        prs: List[float] = []
        for p in csv_paths:
            df = pd.read_csv(p)
            for c in ["PR-AUC", "Ensemble_PR", "PR", "Val PR-AUC", "Ensemble_PR:", "PR_auc", "val_pr"]:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce")
                    prs.extend(s[~s.isna()].tolist())
        if prs:
            return float(np.nanmean(prs))
    except Exception:
        return None
    return None


def fold_key_from_path(path: str) -> int:
    bn = os.path.basename(path)
    for tok in bn.replace(".pt", "").replace("-", "_").split("_"):
        if tok.lower().startswith("fold"):
            try:
                return int(tok.lower().replace("fold", ""))
            except Exception:
                pass
    m = re.findall(r"(\d+)", bn)
    return int(m[-1]) if m else -1


def detect_gatv2_heads_from_sd(sd: Dict[str, torch.Tensor]) -> int:
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and k.endswith("att") and v.dim() == 3:
            return int(v.shape[1])
    return 1


def build_model_for_sd(sd: Dict[str, torch.Tensor], dims: Dict[str, int]) -> nn.Module:
    x_dim, g_dim, r_dim, e_dim = dims["x"], dims["g"], dims["r"], dims["edge"]
    keys = list(sd.keys())
    keys_join = "_".join(keys).lower()
    if any(("g1." in k and "gat" in keys_join) for k in keys):
        heads = detect_gatv2_heads_from_sd(sd)
        return GATv2_DX_DG_DR_H4(x_dim, g_dim, r_dim, e_dim) if heads >= 4 else GATv2_DX_DG_DR(x_dim, g_dim, r_dim, e_dim)
    if any(("c1." in k and "lin_edge" in k) for k in keys):
        return Transformer_DX_DG_DR(x_dim, g_dim, r_dim, e_dim)
    if any(("c1.weight" in k or "n1." in k) for k in keys):
        return NNConv_DX_DG_DR(x_dim, g_dim, r_dim, e_dim)
    return Transformer_DX_DG_DR(x_dim, g_dim, r_dim, e_dim)


def safe_load_compatible_tensors(model: nn.Module, sd: Dict[str, torch.Tensor]) -> nn.Module:
    msd = model.state_dict()
    compatible = {}
    for k, v in sd.items():
        if k in msd and isinstance(v, torch.Tensor) and msd[k].shape == v.shape:
            compatible[k] = v
    model.load_state_dict(compatible, strict=False)
    return model


def predict_family_probs(
    weight_files: Sequence[str],
    dims: Dict[str, int],
    data_list: List[Data],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if not weight_files:
        raise FileNotFoundError("No compatible weight files were found for the requested family.")

    fold_map: Dict[int, List[str]] = {}
    for wf in sorted(set(weight_files)):
        fold_map.setdefault(fold_key_from_path(wf), []).append(wf)

    dl = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    fold_probs: List[np.ndarray] = []

    for _, plist in sorted(fold_map.items()):
        probs_for_fold: List[np.ndarray] = []
        for wt in plist:
            sd = torch.load(wt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model = build_model_for_sd(sd, dims).to(device)
            model = safe_load_compatible_tensors(model, sd).eval()
            pred_blocks: List[torch.Tensor] = []
            with torch.no_grad():
                for bt in dl:
                    bt = bt.to(device)
                    pred_blocks.append(torch.sigmoid(model(bt)).detach().cpu())
            probs_for_fold.append(torch.cat(pred_blocks).numpy())
            del model
        if probs_for_fold:
            fold_probs.append(np.mean(np.stack(probs_for_fold, axis=0), axis=0))

    return np.mean(np.stack(fold_probs, axis=0), axis=0)


def family_definitions(root: Path) -> Dict[str, Dict[str, object]]:
    m2_dir = root / "results_ens_trans_gatv2_scaffold5fold"
    m4_dir = root / "results_lco5_trans_gat_ens_posaug_advanced"
    return {
        "M2": {
            "dir": m2_dir,
            "weights": list_weight_files(m2_dir, ["*gatv2*fold*.pt", "*gat*fold*.pt", "*trans*fold*.pt", "transformer_fold*.pt"]),
            "val_csvs": list_weight_files(m2_dir, ["*cv*results*.csv", "*combined*.csv", "*metrics*.csv"]),
        },
        "M4": {
            "dir": m4_dir,
            "weights": list_weight_files(m4_dir, ["*gatv2*fold*.pt", "*gat*fold*.pt", "*trans*fold*.pt", "transformer_fold*.pt"]),
            "val_csvs": list_weight_files(m4_dir, ["*lco*metrics*.csv", "*cv*.csv", "*results*.csv"]),
        },
    }


# ---------- Main inference ----------
def triage_label(p: float, thr_green: float, thr_red: float) -> str:
    if pd.isna(p):
        return "INVALID"
    if p < thr_green:
        return "GREEN"
    if p < thr_red:
        return "YELLOW"
    return "RED"


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scaler_g_path = root / "scaler_g.joblib"
    scaler_r_path = root / "scaler_r.joblib"
    if not scaler_g_path.exists() or not scaler_r_path.exists():
        raise FileNotFoundError(
            "scaler_g.joblib and scaler_r.joblib were not found under the specified --root directory."
        )

    scaler_g = joblib.load(scaler_g_path)
    scaler_r = joblib.load(scaler_r_path)

    df = pd.read_csv(args.input)
    smiles_col = args.smiles_col or find_col(df, ["SMILES", "smiles", "SMILES_ISO", "smiles_iso"])
    if smiles_col is None:
        raise ValueError("A SMILES column was not found in the input CSV.")
    id_col = args.id_col or find_col(df, ["compound_id", "Compound_ID", "id", "name", "drug"])

    work = df.copy()
    work["compound_id"] = work[id_col].astype(str) if id_col else [f"CMPD_{i+1:04d}" for i in range(len(work))]
    work["smiles_input"] = work[smiles_col].astype(str)
    work["smiles_canonical"] = work["smiles_input"].apply(canonicalize_smiles)
    work["valid_rdkit"] = work["smiles_canonical"].notna()
    work["note"] = np.where(work["valid_rdkit"], "", "RDKit failed to parse the supplied SMILES")

    valid_idx: List[int] = []
    data_list: List[Data] = []
    for idx, row in work.iterrows():
        if not row["valid_rdkit"]:
            continue
        data = mol_to_data(row["smiles_canonical"], row["compound_id"], scaler_g, scaler_r)
        if data is None:
            work.at[idx, "valid_rdkit"] = False
            work.at[idx, "note"] = "Graph construction failed after RDKit parsing"
            continue
        valid_idx.append(idx)
        data_list.append(data)

    out = work[["compound_id", "smiles_input", "smiles_canonical", "valid_rdkit", "note"]].copy()
    out["p_m2"] = np.nan
    out["p_m4"] = np.nan
    out["p_ens"] = np.nan
    out["binary_call_055"] = np.nan
    out["triage_label"] = "INVALID"

    if data_list:
        sample = data_list[0]
        dims = {
            "x": int(sample.x.size(-1)),
            "g": int(sample.g.size(-1)),
            "r": int(sample.r.size(-1)),
            "edge": int(sample.edge_attr.size(-1)) if sample.edge_attr.numel() else 6,
        }

        fams = family_definitions(root)
        p_m2 = predict_family_probs(fams["M2"]["weights"], dims, data_list, args.batch_size, device)
        p_m4 = predict_family_probs(fams["M4"]["weights"], dims, data_list, args.batch_size, device)

        if args.family_weight_mode == "equal":
            w_m2, w_m4 = 0.5, 0.5
        else:
            s_m2 = infer_weights_from_val_csv(fams["M2"]["val_csvs"]) or 1.0
            s_m4 = infer_weights_from_val_csv(fams["M4"]["val_csvs"]) or 1.0
            denom = s_m2 + s_m4
            w_m2 = s_m2 / denom
            w_m4 = s_m4 / denom

        p_ens = (w_m2 * p_m2) + (w_m4 * p_m4)

        valid_pred = pd.DataFrame(
            {
                "row_index": valid_idx,
                "p_m2": p_m2,
                "p_m4": p_m4,
                "p_ens": p_ens,
            }
        )
        for _, r in valid_pred.iterrows():
            i = int(r["row_index"])
            p = float(r["p_ens"])
            out.at[i, "p_m2"] = float(r["p_m2"])
            out.at[i, "p_m4"] = float(r["p_m4"])
            out.at[i, "p_ens"] = p
            out.at[i, "binary_call_055"] = int(p >= args.binary_thr)
            out.at[i, "triage_label"] = triage_label(p, args.thr_green, args.thr_red)

    out.to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}")


if __name__ == "__main__":
    main()
