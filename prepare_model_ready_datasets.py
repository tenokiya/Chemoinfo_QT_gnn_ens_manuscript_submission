#!/usr/bin/env python3
"""
Prepare model-ready dataset files for the QT-M2M4 manuscript workflow.

Purpose
-------
This utility provides command-line entry points that make the dataset assembly
steps more explicit for readers who want to understand or regenerate the
formatted CSV / graph-index files used in the manuscript workflow.

Scope
-----
The manuscript repository already includes the canonical processed files used in
analysis and evaluation. This script is an auxiliary transparency tool that
helps readers move from public-source assay tables (and, when available,
auxiliary positive lists) to model-ready CSV/index artifacts.

Supported tasks
---------------
1) label-aid1671200
   Convert the raw PubChem AID 1671200 table into labeled AID-only tables,
   including the mixed internal AID table used before FAERS-derived positives
   are merged in.

2) merge-internal
   Merge the AID-only mixed table with an auxiliary positive table
   (for example, a FAERS-derived positive list) and create the consolidated
   internal training table.

3) build-graph-index
   Convert a consolidated internal table into a graph-index CSV and,
   optionally, a PyTorch Geometric .pt object list.

4) build-external
   Convert the raw PubChem AID 588834 table into a labeled external table,
   graph-index CSV, and, optionally, a PyTorch Geometric .pt object list.

Notes
-----
- This script aims to expose the preparation logic in a reusable command-line
  form. It is not presented as the sole authoritative source of the manuscript
  artifacts; the shipped processed files remain the canonical review assets.
- When graph .pt generation is requested, the script uses the same descriptor /
  graph feature scheme used in predict_qt_liability.py.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdchem

RDLogger.DisableLog("rdApp.*")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare model-ready QT manuscript dataset files")
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("label-aid1671200", help="Create labeled AID 1671200 internal tables")
    p1.add_argument("--input", required=True, help="Raw AID_1671200_datatable.csv")
    p1.add_argument("--outdir", default=".", help="Directory for output files")
    p1.add_argument("--min_activity_score", type=float, default=40.0)
    p1.add_argument("--min_abs_max_response", type=float, default=10.0)
    p1.add_argument("--require_fit_logac50", action="store_true", default=True)
    p1.add_argument("--no_require_fit_logac50", dest="require_fit_logac50", action="store_false")
    p1.add_argument("--exclude_alerts", action="store_true", default=True)
    p1.add_argument("--keep_alerts", dest="exclude_alerts", action="store_false")

    p2 = sub.add_parser("merge-internal", help="Merge AID-only mixed table with auxiliary positives")
    p2.add_argument("--aid_mix", required=True, help="AID_1671200_labeled_MIX_POSstrict_NEGbase.csv")
    p2.add_argument(
        "--aux_positive_csv",
        required=True,
        help="Auxiliary positive table (for example, FAERS-derived positives) with at least CID/SMILES",
    )
    p2.add_argument("--outdir", default=".", help="Directory for output files")

    p3 = sub.add_parser("build-graph-index", help="Build graph-index CSV / optional .pt from a consolidated table")
    p3.add_argument("--input", required=True, help="Consolidated internal table CSV")
    p3.add_argument("--output_index", required=True, help="Output graph-index CSV")
    p3.add_argument("--write_pt", default=None, help="Optional output .pt path")
    p3.add_argument("--root", default=".", help="Repository root containing scaler_g.joblib and scaler_r.joblib")
    p3.add_argument("--pos_aug", type=int, default=6, help="Target number of SMILES variants per positive entry")
    p3.add_argument("--neg_aug", type=int, default=1, help="Target number of SMILES variants per negative entry")
    p3.add_argument("--seed", type=int, default=42)

    p4 = sub.add_parser("build-external", help="Build labeled / graph-index files from raw AID 588834")
    p4.add_argument("--input", required=True, help="Raw AID_588834_datatable.csv")
    p4.add_argument("--outdir", default=".", help="Directory for output files")
    p4.add_argument("--write_pt", default=None, help="Optional output .pt path")
    p4.add_argument("--root", default=".", help="Repository root containing scaler_g.joblib and scaler_r.joblib")

    return parser.parse_args()


# ---------- Shared helpers ----------
def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def write_json(path: str | os.PathLike[str], obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def numeric_rows(df: pd.DataFrame, cid_col: str = "PUBCHEM_CID") -> pd.DataFrame:
    out = df.copy()
    out[cid_col] = pd.to_numeric(out[cid_col], errors="coerce")
    out = out[out[cid_col].notna()].copy()
    out[cid_col] = out[cid_col].astype("Int64")
    return out


def choose_smiles_column(df: pd.DataFrame) -> Optional[str]:
    for c in ["PUBCHEM_EXT_DATASOURCE_SMILES", "CanonicalSMILES", "SMILES", "IsomericSMILES", "smiles"]:
        if c in df.columns:
            return c
    return None


def canonicalize_smiles(smiles: str) -> Optional[str]:
    if pd.isna(smiles):
        return None
    s = str(smiles).strip()
    if not s or s.lower() in {"nan", "none"}:
        return None
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def random_smiles_variants(smiles: str, n: int, seed: int) -> List[str]:
    can = canonicalize_smiles(smiles)
    if can is None:
        return []
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return []
    rs = np.random.RandomState(seed)
    out = [can]
    for _ in range(max(n * 10, 20)):
        if len(out) >= n:
            break
        try:
            smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
        except Exception:
            smi = None
        if smi and smi not in out:
            out.append(smi)
    while len(out) < n:
        out.append(can)
    return out[:n]


# ---------- AID 1671200 helpers ----------
def phenotype_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.match(r"^Phenotype(-Replicate_\d+)?$", c)]


def curve_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.search(r"Curve_Description", c, re.I)]


def fit_logac50_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.search(r"Fit_LogAC50", c, re.I)]


def max_response_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if re.search(r"Max_Response", c, re.I)]


def any_inhibitor(df: pd.DataFrame) -> pd.Series:
    pcs = phenotype_cols(df)
    if not pcs:
        return pd.Series(False, index=df.index)
    txt = df[pcs].astype(str).apply(lambda r: " ".join(r.values.tolist()).lower(), axis=1)
    return txt.str.contains("inhibitor", regex=False).fillna(False)


def any_full_or_partial(df: pd.DataFrame) -> pd.Series:
    ccs = curve_cols(df)
    if not ccs:
        return pd.Series(True, index=df.index)
    txt = df[ccs].astype(str).apply(lambda r: " ".join(r.values.tolist()).lower(), axis=1)
    return (txt.str.contains("full") | txt.str.contains("partial")).fillna(False)


def any_fit_logac50(df: pd.DataFrame) -> pd.Series:
    fcs = fit_logac50_cols(df)
    if not fcs:
        return pd.Series(True, index=df.index)
    x = df[fcs].apply(pd.to_numeric, errors="coerce")
    return x.notna().any(axis=1)


def max_abs_response(df: pd.DataFrame) -> pd.Series:
    mcs = max_response_cols(df)
    if not mcs:
        return pd.Series(np.nan, index=df.index)
    x = df[mcs].apply(pd.to_numeric, errors="coerce").abs()
    return x.max(axis=1)


def structural_alert_mask(smiles_series: pd.Series) -> pd.Series:
    try:
        from rdkit.Chem import FilterCatalog, FilterCatalogParams
    except Exception:
        return pd.Series(False, index=smiles_series.index)

    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.NIH)
    catalog = FilterCatalog.FilterCatalog(params)

    mask = []
    for s in smiles_series.fillna(""):
        mol = Chem.MolFromSmiles(str(s))
        mask.append(bool(mol is not None and catalog.HasMatch(mol)))
    return pd.Series(mask, index=smiles_series.index)


def cmd_label_aid1671200(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)
    raw = pd.read_csv(args.input, low_memory=False)
    df = numeric_rows(raw)

    smiles_col = choose_smiles_column(df)
    if smiles_col is None:
        raise SystemExit("No SMILES column was found in the AID 1671200 table.")

    outcome = df["PUBCHEM_ACTIVITY_OUTCOME"].astype(str).str.strip().str.upper()
    inhibitor = any_inhibitor(df)

    pos = (outcome == "ACTIVE") & inhibitor
    neg = outcome == "INACTIVE"

    base = df[pos | neg].copy()
    base["label_herg_inhibit"] = np.where(pos.loc[base.index], 1, 0)
    base["qc_pass"] = True

    score = pd.to_numeric(base.get("PUBCHEM_ACTIVITY_SCORE"), errors="coerce") if "PUBCHEM_ACTIVITY_SCORE" in base.columns else pd.Series(np.nan, index=base.index)
    fit_ok = any_fit_logac50(base)
    curve_ok = any_full_or_partial(base)
    resp_ok = max_abs_response(base).fillna(np.inf) >= args.min_abs_max_response
    score_ok = score.fillna(-np.inf) >= args.min_activity_score

    qc_positive = (base["label_herg_inhibit"] == 1) & score_ok & curve_ok & resp_ok
    if args.require_fit_logac50:
        qc_positive &= fit_ok

    qc = base[(base["label_herg_inhibit"] == 0) | qc_positive].copy()
    qc["qc_pass"] = np.where(qc["label_herg_inhibit"] == 0, True, qc_positive.loc[qc.index])

    qc_np = qc.copy()
    if args.exclude_alerts:
        alert = structural_alert_mask(qc_np[smiles_col])
        qc_np = qc_np[(qc_np["label_herg_inhibit"] == 0) | (~alert)].copy()
        qc_np["structural_alert"] = alert.loc[qc_np.index]
    else:
        qc_np["structural_alert"] = False

    keep_cols = [c for c in ["PUBCHEM_CID", smiles_col, "label_herg_inhibit"] if c in qc_np.columns]
    pos_strict = qc_np[qc_np["label_herg_inhibit"] == 1][keep_cols].drop_duplicates()
    neg_base = base[base["label_herg_inhibit"] == 0][keep_cols].drop_duplicates()
    if smiles_col in keep_cols:
        neg_base = neg_base[~neg_base[smiles_col].isin(set(pos_strict[smiles_col].dropna()))]
    mix = pd.concat([pos_strict, neg_base], ignore_index=True).drop_duplicates()

    out_base = Path(args.outdir) / "AID_1671200_labeled.csv"
    out_qc = Path(args.outdir) / "AID_1671200_labeled_QC.csv"
    out_qc_np = Path(args.outdir) / "AID_1671200_labeled_QC_noPAINS.csv"
    out_mix = Path(args.outdir) / "AID_1671200_labeled_MIX_POSstrict_NEGbase.csv"
    out_audit = Path(args.outdir) / "AID_1671200_label_audit.json"

    base.to_csv(out_base, index=False)
    qc.to_csv(out_qc, index=False)
    qc_np.to_csv(out_qc_np, index=False)
    mix.to_csv(out_mix, index=False)

    audit = {
        "raw_numeric_rows": int(len(df)),
        "base_rows": int(len(base)),
        "base_label_counts": {str(k): int(v) for k, v in base["label_herg_inhibit"].value_counts().to_dict().items()},
        "qc_rows": int(len(qc)),
        "qc_no_alert_rows": int(len(qc_np)),
        "mix_rows": int(len(mix)),
        "mix_label_counts": {str(k): int(v) for k, v in mix["label_herg_inhibit"].value_counts().to_dict().items()},
        "smiles_column": smiles_col,
    }
    write_json(out_audit, audit)
    print(f"Saved: {out_base}")
    print(f"Saved: {out_qc}")
    print(f"Saved: {out_qc_np}")
    print(f"Saved: {out_mix}")
    print(f"Saved: {out_audit}")


# ---------- Merge internal helpers ----------
def normalize_aux_positive(df: pd.DataFrame) -> pd.DataFrame:
    cid_col = None
    for c in ["PUBCHEM_CID", "CID", "pubchem_cid"]:
        if c in df.columns:
            cid_col = c
            break
    smiles_col = choose_smiles_column(df)
    if smiles_col is None:
        raise SystemExit("Auxiliary positive CSV must contain a SMILES-like column.")

    out = pd.DataFrame()
    out["PUBCHEM_CID"] = pd.to_numeric(df[cid_col], errors="coerce").astype("Int64") if cid_col else pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))
    out["SMILES"] = df[smiles_col].astype(str).str.strip()
    out.loc[out["SMILES"].isin(["", "nan", "None"]), "SMILES"] = pd.NA
    if "signal" in df.columns:
        signal = pd.to_numeric(df["signal"], errors="coerce")
        out = out[signal.fillna(1) == 1].copy()
    out["drug"] = df["drug"] if "drug" in df.columns else pd.NA
    out["source"] = df["source"] if "source" in df.columns else "aux_positive"
    out["label_herg_inhibit"] = 1
    out["label"] = 1
    return out.dropna(subset=["SMILES"]).drop_duplicates()


def cmd_merge_internal(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)
    aid = pd.read_csv(args.aid_mix, low_memory=False)
    aux = pd.read_csv(args.aux_positive_csv, low_memory=False)

    aid_smiles_col = choose_smiles_column(aid)
    if aid_smiles_col is None:
        raise SystemExit("AID mixed table must contain a SMILES-like column.")

    aid_min = pd.DataFrame()
    aid_min["PUBCHEM_CID"] = pd.to_numeric(aid["PUBCHEM_CID"], errors="coerce").astype("Int64") if "PUBCHEM_CID" in aid.columns else pd.Series(pd.array([pd.NA] * len(aid), dtype="Int64"))
    aid_min["SMILES"] = aid[aid_smiles_col].astype(str).str.strip()
    aid_min["signal"] = pd.NA
    aid_min["drug"] = pd.NA
    aid_min["source"] = "AID1671200"
    aid_min["label_herg_inhibit"] = pd.to_numeric(aid.get("label_herg_inhibit"), errors="coerce")
    aid_min["label"] = aid_min["label_herg_inhibit"]
    aid_min = aid_min.dropna(subset=["SMILES", "label"]).drop_duplicates()

    aux_norm = normalize_aux_positive(aux)

    aid_excluded = aid_min[~aid_min["SMILES"].isin(set(aux_norm["SMILES"].dropna()))].copy()
    merged = pd.concat([aux_norm, aid_excluded], ignore_index=True).drop_duplicates()

    # Consolidate by canonical parent key
    merged["Parent_Key"] = merged["SMILES"].map(canonicalize_smiles)
    merged = merged.dropna(subset=["Parent_Key"]).copy()
    merged["PUBCHEM_CID"] = pd.to_numeric(merged["PUBCHEM_CID"], errors="coerce").astype("Int64")
    merged["label"] = pd.to_numeric(merged["label"], errors="coerce").astype(int)

    rows = []
    for parent_key, grp in merged.groupby("Parent_Key", dropna=False):
        grp = grp.copy()
        rep = grp.iloc[0]
        rows.append(
            {
                "SMILES": parent_key,
                "Parent_Key": parent_key,
                "PUBCHEM_CID": rep["PUBCHEM_CID"],
                "source_rep": rep.get("source", pd.NA),
                "label": int(grp["label"].max()),
                "Sample_Weight": 1.0,
                "Children_N": int(len(grp)),
            }
        )
    consolidated = pd.DataFrame(rows)

    out_aid = Path(args.outdir) / "aid_minimal.csv"
    out_aux = Path(args.outdir) / "aux_positive_minimal.csv"
    out_merged = Path(args.outdir) / "merged_for_model_with_label.csv"
    out_cons = Path(args.outdir) / "merged_for_model_consolidated.csv"
    out_audit = Path(args.outdir) / "merged_for_model_audit.json"

    aid_excluded.to_csv(out_aid, index=False)
    aux_norm.to_csv(out_aux, index=False)
    merged.to_csv(out_merged, index=False)
    consolidated.to_csv(out_cons, index=False)
    write_json(
        out_audit,
        {
            "aid_rows_after_aux_smiles_exclusion": int(len(aid_excluded)),
            "aux_positive_rows": int(len(aux_norm)),
            "merged_rows": int(len(merged)),
            "consolidated_rows": int(len(consolidated)),
            "consolidated_label_counts": {str(k): int(v) for k, v in consolidated["label"].value_counts().to_dict().items()},
        },
    )
    print(f"Saved: {out_aid}")
    print(f"Saved: {out_aux}")
    print(f"Saved: {out_merged}")
    print(f"Saved: {out_cons}")
    print(f"Saved: {out_audit}")


# ---------- Graph feature helpers ----------
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


def atom_f(atom: rdchem.Atom):
    import torch

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


def bond_f(bond: rdchem.Bond):
    import torch

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


def build_graph_pt(index_df: pd.DataFrame, write_pt: str, root: str) -> None:
    import joblib
    import torch
    from sklearn.preprocessing import StandardScaler

    try:
        from torch_geometric.data import Data
    except Exception as e:
        raise SystemExit(
            "torch-geometric is required when --write_pt is used. "
            "Please install requirements.txt and retry.\n"
            f"Original import error: {e}"
        )

    scaler_g_path = Path(root) / "scaler_g.joblib"
    scaler_r_path = Path(root) / "scaler_r.joblib"

    mols = []
    g_raw = []
    r_raw = []
    for smi in index_df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            mols.append(None)
            g_raw.append([np.nan] * 7)
            r_raw.append([np.nan] * 10)
            continue
        mols.append(mol)
        g_raw.append(gfeat7(mol))
        r_raw.append(rdesc10(mol))

    g_raw = np.asarray(g_raw, dtype=float)
    r_raw = np.asarray(r_raw, dtype=float)
    good = np.isfinite(g_raw).all(axis=1) & np.isfinite(r_raw).all(axis=1)
    index_df = index_df.loc[good].reset_index(drop=True)
    mols = [m for m, k in zip(mols, good) if k]
    g_raw = g_raw[good]
    r_raw = r_raw[good]

    if scaler_g_path.exists() and scaler_r_path.exists():
        scaler_g = joblib.load(scaler_g_path)
        scaler_r = joblib.load(scaler_r_path)
    else:
        scaler_g = StandardScaler().fit(g_raw)
        scaler_r = StandardScaler().fit(r_raw)
        joblib.dump(scaler_g, scaler_g_path)
        joblib.dump(scaler_r, scaler_r_path)

    g_scaled = scaler_g.transform(g_raw)
    r_scaled = scaler_r.transform(r_raw)

    data_list = []
    for i, (mol, (_, row)) in enumerate(zip(mols, index_df.iterrows())):
        import torch

        atoms = [atom_f(a) for a in mol.GetAtoms()]
        if not atoms:
            continue
        x = torch.stack(atoms)

        edge_pairs = []
        edge_attrs = []
        for bond in mol.GetBonds():
            a = bond.GetBeginAtomIdx()
            b = bond.GetEndAtomIdx()
            bf = bond_f(bond)
            edge_pairs.extend([[a, b], [b, a]])
            edge_attrs.extend([bf, bf])
        if edge_pairs:
            edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_attrs)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 6), dtype=torch.float)

        d = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            g=torch.tensor(g_scaled[i:i + 1], dtype=torch.float),
            r=torch.tensor(r_scaled[i:i + 1], dtype=torch.float),
            y=torch.tensor([int(row["Label"])], dtype=torch.long),
        )
        d.cid = int(row["CID"]) if pd.notna(row["CID"]) else -1
        d.smiles_input = str(row["SMILES"])
        data_list.append(d)

    torch.save(data_list, write_pt)



def cmd_build_graph_index(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.input, low_memory=False)
    smiles_col = choose_smiles_column(df)
    if smiles_col is None:
        raise SystemExit("Input consolidated table must contain a SMILES-like column.")

    label_col = "label" if "label" in df.columns else "Label"
    cid_col = "PUBCHEM_CID" if "PUBCHEM_CID" in df.columns else "CID"
    wt_col = "Sample_Weight" if "Sample_Weight" in df.columns else None

    rng = np.random.RandomState(args.seed)
    rows = []
    for _, row in df.iterrows():
        smi = row[smiles_col]
        label = int(row[label_col])
        cid = row[cid_col] if cid_col in row else pd.NA
        wt = float(row[wt_col]) if wt_col and pd.notna(row[wt_col]) else 1.0
        n = args.pos_aug if label == 1 else args.neg_aug
        seed = int(rng.randint(0, 1_000_000))
        for var in random_smiles_variants(str(smi), n=n, seed=seed):
            rows.append({"SMILES": var, "Label": label, "SampleWeight": wt, "CID": cid})

    index_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    index_df.to_csv(args.output_index, index=False)
    print(f"Saved: {args.output_index}")

    if args.write_pt:
        build_graph_pt(index_df, args.write_pt, args.root)
        print(f"Saved: {args.write_pt}")


# ---------- External helpers ----------
def cmd_build_external(args: argparse.Namespace) -> None:
    ensure_dir(args.outdir)
    raw = pd.read_csv(args.input, low_memory=False)
    df = numeric_rows(raw)
    smiles_col = choose_smiles_column(df)
    if smiles_col is None:
        raise SystemExit("No SMILES column was found in the AID 588834 table.")

    outcome = df["PUBCHEM_ACTIVITY_OUTCOME"].astype(str).str.strip().str.upper()
    label_map = {"ACTIVE": 1, "INACTIVE": 0}
    df["label_herg_inhibit"] = outcome.map(label_map)
    lab = df[df["label_herg_inhibit"].notna()].copy()
    lab = lab.dropna(subset=[smiles_col]).copy()
    lab[smiles_col] = lab[smiles_col].astype(str).str.strip()
    lab = lab[lab[smiles_col].ne("")].copy()
    lab["SMILES"] = lab[smiles_col]
    lab["label"] = lab["label_herg_inhibit"].astype(int)

    out_labeled = Path(args.outdir) / "AID_588834_labeled.csv"
    out_index = Path(args.outdir) / "data_graph_external_index.csv"
    out_audit = Path(args.outdir) / "AID_588834_label_audit.json"

    lab.to_csv(out_labeled, index=False)
    index_df = lab[["SMILES", "label", "PUBCHEM_CID"]].copy()
    index_df.columns = ["SMILES", "Label", "CID"]
    index_df["SampleWeight"] = 1.0
    index_df = index_df[["SMILES", "Label", "SampleWeight", "CID"]].drop_duplicates().reset_index(drop=True)
    index_df.to_csv(out_index, index=False)

    if args.write_pt:
        build_graph_pt(index_df, args.write_pt, args.root)
        print(f"Saved: {args.write_pt}")

    write_json(
        out_audit,
        {
            "raw_numeric_rows": int(len(df)),
            "labeled_rows": int(len(lab)),
            "label_counts": {str(k): int(v) for k, v in lab["label"].value_counts().to_dict().items()},
            "graph_index_rows": int(len(index_df)),
            "smiles_column": smiles_col,
        },
    )
    print(f"Saved: {out_labeled}")
    print(f"Saved: {out_index}")
    print(f"Saved: {out_audit}")


def main() -> None:
    args = parse_args()
    if args.command == "label-aid1671200":
        cmd_label_aid1671200(args)
    elif args.command == "merge-internal":
        cmd_merge_internal(args)
    elif args.command == "build-graph-index":
        cmd_build_graph_index(args)
    elif args.command == "build-external":
        cmd_build_external(args)
    else:
        raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
