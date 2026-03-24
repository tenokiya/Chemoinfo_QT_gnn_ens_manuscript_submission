# ================================================
# Strict External Evaluation (Chemoinfo_QT)
# - Automatically generate a strict external dataset (excluding duplicates of CID/standard SMILES)
# - Create a strict external .pt file
# - Evaluates rigorous external data using M1..M4 (automatic search of gatv2/trans/nn)
# - Outputs comparisons and rankings of all combinations, as well as rigorous ROC, PR, and confusion rates for the best combination
# ================================================

Translated with DeepL.com (free version)
import os, re, glob, itertools, warnings, json, math
from pathlib import Path

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 12345
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH  = 256
P_AT_K = [10, 25, 50, 100]
FIXED_DECISION_THRESHOLD = 0.55

# plot layout matched to Figure2c/Figure2d (2048 x 1436 px)
PLOT_WIDTH_PX = 2048
PLOT_HEIGHT_PX = 1436
PLOT_DPI = 200
PLOT_FIGSIZE = (PLOT_WIDTH_PX / PLOT_DPI, PLOT_HEIGHT_PX / PLOT_DPI)
PLOT_LEFT = 0.10
PLOT_RIGHT = 0.98
PLOT_BOTTOM = 0.12
PLOT_TOP = 0.98
PLOT_ROC_LEGEND_LOC = "upper left"
PLOT_PR_LEGEND_LOC = "upper right"
CURVE_LINEWIDTH = 2.5
DIAG_LINEWIDTH = 2.5
AXIS_LABELSIZE = 18
TICK_LABELSIZE = 16
LEGEND_FONTSIZE = 18
SPINE_LINEWIDTH = 1.2

# ---------- User settings ----------
ROOT = "/content/drive/MyDrive/Chemoinfo_QT"

# internal / external resources
INTERNAL_CURATED_CSV = f"{ROOT}/merged_for_model_consolidated.csv"
EXTERNAL_COMPOUND_CSV = f"{ROOT}/external_AID_588834_herg.csv"
PT_INTERNAL = f"{ROOT}/data_graph_with_smiles.pt"
PT_EXTERNAL = f"{ROOT}/data_graph_external.pt"
IDX_INTERNAL = f"{ROOT}/data_graph_with_smiles_index.csv"
IDX_EXTERNAL = f"{ROOT}/data_graph_external_index.csv"

# strict rule
STRICT_OVERLAP_RULE = "cid_or_canonical"   # "cid_or_canonical" / "cid_only" / "canonical_only"
SAVE_STRICT_PT = True

# strict output dataset
STRICT_EXTERNAL_ANNOT = f"{ROOT}/external_graph_overlap_annotated.csv"
STRICT_EXTERNAL_CSV   = f"{ROOT}/external_graph_strict.csv"
STRICT_EXTERNAL_PT    = f"{ROOT}/data_graph_external_strict.pt"

# family directories
M1_DIR = f"{ROOT}/results_ens_trans_gatv2_5fold"
M2_DIR = f"{ROOT}/results_ens_trans_gatv2_scaffold5fold"
M3_DIR = f"{ROOT}/results_lco5_trans_gatv2_ens_posaug"
M4_DIR = f"{ROOT}/results_lco5_trans_gat_ens_posaug_advanced"

# outputs
OUT_ALL   = f"{ROOT}/strict_external_eval_comparison_all.csv"
OUT_RANK  = f"{ROOT}/strict_external_eval_ranking.csv"
OUT_BESTP = f"{ROOT}/strict_external_best_predictions.csv"
OUT_METR  = f"{ROOT}/strict_external_best_metrics.json"
OUT_CONF  = f"{ROOT}/strict_external_best_confusion_counts.csv"
OUT_ROC   = f"{ROOT}/strict_external_best_roc_curve.png"
OUT_PR    = f"{ROOT}/strict_external_best_pr_curve.png"
OUT_ROCPT = f"{ROOT}/strict_external_best_roc_curve_points.csv"
OUT_PRPT  = f"{ROOT}/strict_external_best_pr_curve_points.csv"

os.makedirs(ROOT, exist_ok=True)

# ---------- Imports requiring optional install ----------
def ensure_pkg(pkg_name, import_name=None):
    import importlib, subprocess, sys
    name = import_name or pkg_name
    try:
        return importlib.import_module(name)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return importlib.import_module(name)

mpl = ensure_pkg("matplotlib", "matplotlib")
import matplotlib.pyplot as plt

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:
    ensure_pkg("rdkit")
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

try:
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    from torch_geometric.nn import TransformerConv, NNConv, AttentionalAggregation, GATv2Conv
    from torch.serialization import add_safe_globals
except Exception:
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch-geometric"])
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
    from torch_geometric.nn import TransformerConv, NNConv, AttentionalAggregation, GATv2Conv
    from torch.serialization import add_safe_globals

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score, roc_curve,
    confusion_matrix
)
from rdkit.Chem import AllChem, DataStructs

# ---------- Helpers ----------
def log(msg): print(msg, flush=True)

def read_csv_flex(path):
    if not Path(path).exists():
        return None
    for enc in ["utf-8", "utf-8-sig", "cp932", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def find_col(df, candidates):
    if df is None or df.empty:
        return None
    norm = {str(c).strip().lower(): c for c in df.columns}
    # exact
    for cand in candidates:
        c = cand.strip().lower()
        if c in norm:
            return norm[c]
    # contains
    for cand in candidates:
        c = cand.strip().lower()
        for k, orig in norm.items():
            if c in k:
                return orig
    return None

def parse_cid(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return np.nan
    m = re.search(r"(\d+)", s)
    if not m:
        return np.nan
    try:
        return int(m.group(1))
    except Exception:
        return np.nan

def canonicalize_smiles(smi):
    if pd.isna(smi):
        return np.nan
    smi = str(smi).strip()
    if smi == "" or smi.lower() in {"nan", "none"}:
        return np.nan
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.nan
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return np.nan

def murcko_smiles(smi):
    if pd.isna(smi):
        return np.nan
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return np.nan
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return np.nan

def fp_from_smiles(smi, radius=2, nbits=2048):
    if pd.isna(smi):
        return None
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)

def tanimoto_to_set(fp, fp_list):
    if fp is None or len(fp_list) == 0:
        return np.nan
    sims = DataStructs.BulkTanimotoSimilarity(fp, fp_list)
    return float(max(sims)) if len(sims) else np.nan

def choose_metadata_table(df, source_name):
    if df is None or df.empty:
        raise RuntimeError(f"{source_name}: table not found or empty")
    out = df.copy()
    cid_col = find_col(out, ["cid", "pubchem_cid", "compound_cid"])
    smi_col = find_col(out, ["canonical_smiles", "smiles", "resolved_smiles", "standardized_smiles", "input_smiles"])
    label_col = find_col(out, ["y", "label", "class", "target"])
    tier_col = find_col(out, ["tier", "activity_tier"])
    if cid_col:
        out["cid_norm"] = out[cid_col].map(parse_cid)
    else:
        out["cid_norm"] = np.nan
    if smi_col:
        out["smiles_used"] = out[smi_col].astype(str)
    else:
        out["smiles_used"] = np.nan
    out["canonical_smiles"] = out["smiles_used"].map(canonicalize_smiles)
    if label_col:
        out["label_norm"] = pd.to_numeric(out[label_col], errors="coerce")
    else:
        out["label_norm"] = np.nan
    if tier_col:
        out["tier_norm"] = out[tier_col].astype(str).str.lower()
    else:
        out["tier_norm"] = ""
    out["source_table"] = source_name
    return out

# ---------- PyG safe load ----------
def load_pyg_list(path: str):
    try:
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage
        add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
    except Exception as e:
        log(f"[WARN] add_safe_globals failed: {e}")
    try:
        obj = torch.load(path, map_location="cpu")
    except Exception:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "data_list" in obj:
        return obj["data_list"]
    try:
        return list(obj)
    except Exception as e:
        raise RuntimeError(f"Unknown .pt format ({type(obj)}): {e}")

# ---------- Feature-dim inference ----------
def infer_dims(data_list):
    for d in data_list:
        if hasattr(d, "x") and hasattr(d, "edge_attr") and hasattr(d, "g") and hasattr(d, "r"):
            return {
                "x": d.x.size(-1),
                "edge": d.edge_attr.size(-1),
                "g": (d.g.size(-1) if d.g.dim() in (2, 3) else d.g.numel()),
                "r": (d.r.size(-1) if d.r.dim() in (2, 3) else d.r.numel()),
            }
    raise RuntimeError("infer_dims failed (x/edge_attr/g/r が見つかりません)")

# ---------- Models ----------
def make_edge_mlp(in_attr_dim, out_channels, in_channels):
    hidden = max(64, min(256, in_attr_dim * 8))
    return nn.Sequential(
        nn.Linear(in_attr_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, in_channels * out_channels)
    )

class Transformer_DX_DG_DR(nn.Module):
    def __init__(self, X_DIM, G_DIM, R_DIM, E_DIM, HID=256):
        super().__init__()
        self.c1 = TransformerConv(X_DIM, HID, heads=4, concat=False, edge_dim=E_DIM, dropout=0.0, beta=False)
        self.c2 = TransformerConv(HID,   HID, heads=4, concat=False, edge_dim=E_DIM, dropout=0.0, beta=False)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(HID, 1))
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(HID + G_DIM + R_DIM, 1)
    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index, d.edge_attr))
        x = F.relu(self.c2(x,   d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim()==3 else d.g
        r = d.r.squeeze(1) if d.r.dim()==3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)

class NNConv_DX_DG_DR(nn.Module):
    def __init__(self, X_DIM, G_DIM, R_DIM, E_DIM, HID=256):
        super().__init__()
        edge_nn1 = make_edge_mlp(E_DIM, HID, X_DIM)
        edge_nn2 = make_edge_mlp(E_DIM, HID, HID)
        self.c1 = NNConv(X_DIM, HID, edge_nn1, aggr='mean')
        self.c2 = NNConv(HID,  HID, edge_nn2, aggr='mean')
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(HID, 1))
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(HID + G_DIM + R_DIM, 1)
    def forward(self, d):
        x = F.relu(self.c1(d.x, d.edge_index, d.edge_attr))
        x = F.relu(self.c2(x,   d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim()==3 else d.g
        r = d.r.squeeze(1) if d.r.dim()==3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)

class GATv2_DX_DG_DR(nn.Module):
    def __init__(self, X_DIM, G_DIM, R_DIM, E_DIM, HID=256):
        super().__init__()
        self.g1 = GATv2Conv(X_DIM, HID, edge_dim=E_DIM, dropout=0.0)
        self.g2 = GATv2Conv(HID,   HID, edge_dim=E_DIM, dropout=0.0)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(HID, 1))
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(HID + G_DIM + R_DIM, 1)
    def forward(self, d):
        x = F.elu(self.g1(d.x, d.edge_index, d.edge_attr))
        x = F.elu(self.g2(x,   d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim()==3 else d.g
        r = d.r.squeeze(1) if d.r.dim()==3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)

class GATv2_DX_DG_DR_H4(nn.Module):
    def __init__(self, X_DIM, G_DIM, R_DIM, E_DIM, HID=256):
        super().__init__()
        self.g1 = GATv2Conv(X_DIM, HID, heads=4, concat=True, edge_dim=E_DIM, dropout=0.0)
        self.g2 = GATv2Conv(HID*4, HID, heads=4, concat=True, edge_dim=E_DIM, dropout=0.0)
        self.pool = AttentionalAggregation(gate_nn=nn.Linear(HID*4, 1))
        self.drop = nn.Dropout(0.2)
        self.fc   = nn.Linear(HID*4 + G_DIM + R_DIM, 1)
    def forward(self, d):
        x = F.elu(self.g1(d.x, d.edge_index, d.edge_attr))
        x = F.elu(self.g2(x,   d.edge_index, d.edge_attr))
        x = self.pool(self.drop(x), d.batch)
        g = d.g.squeeze(1) if d.g.dim()==3 else d.g
        r = d.r.squeeze(1) if d.r.dim()==3 else d.r
        return self.fc(torch.cat([x, g, r], dim=1)).view(-1)

# ---------- Families & weights ----------
def list_weight_files(folder, patterns):
    files = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(folder, pat)))
    return files

def family_defs():
    return {
        "M1": {
            "dir": M1_DIR,
            "gatv2": list_weight_files(M1_DIR, ["*gatv2*fold*.pt", "*gat*fold*.pt"]),
            "trans": list_weight_files(M1_DIR, ["*trans*fold*.pt", "transformer_fold*.pt"]),
            "nn":    list_weight_files(M1_DIR, ["*nn*fold*.pt", "nnconv_fold*.pt"]),
            "val_csvs": list_weight_files(M1_DIR, ["*cv*results*.csv","*combined*.csv","*metrics*.csv"])
        },
        "M2": {
            "dir": M2_DIR,
            "gatv2": list_weight_files(M2_DIR, ["*gatv2*fold*.pt", "*gat*fold*.pt"]),
            "trans": list_weight_files(M2_DIR, ["*trans*fold*.pt","transformer_fold*.pt"]),
            "nn":    list_weight_files(M2_DIR, ["*nn*fold*.pt","nnconv_fold*.pt"]),
            "val_csvs": list_weight_files(M2_DIR, ["*cv*results*.csv","*combined*.csv","*metrics*.csv"])
        },
        "M3": {
            "dir": M3_DIR,
            "gatv2": list_weight_files(M3_DIR, ["*gatv2*fold*.pt", "*gat*fold*.pt"]),
            "trans": list_weight_files(M3_DIR, ["*trans*fold*.pt","transformer_fold*.pt","trans_fold*.pt"]),
            "nn":    list_weight_files(M3_DIR, ["*nn*fold*.pt","nnconv_fold*.pt"]),
            "val_csvs": list_weight_files(M3_DIR, ["*lco*metrics*.csv","*ensemble*.csv","*cv*.csv"])
        },
        "M4": {
            "dir": M4_DIR,
            "gatv2": list_weight_files(M4_DIR, ["*gatv2*fold*.pt", "*gat*fold*.pt"]),
            "trans": list_weight_files(M4_DIR, ["*trans*fold*.pt","transformer_fold*.pt"]),
            "nn":    list_weight_files(M4_DIR, ["*nn*fold*.pt","nnconv_fold*.pt"]),
            "val_csvs": list_weight_files(M4_DIR, ["*lco*metrics*.csv","*cv*.csv","*results*.csv"])
        },
    }

def infer_weights_from_val_csv(csv_paths):
    if not csv_paths: return None
    try:
        prs = []
        for p in csv_paths:
            df = pd.read_csv(p)
            for c in ["PR-AUC","Ensemble_PR","PR","Val PR-AUC","Ensemble_PR:","PR_auc","val_pr"]:
                if c in df.columns:
                    s = pd.to_numeric(df[c], errors="coerce")
                    prs += s[~s.isna()].tolist()
        if prs:
            return float(np.nanmean(prs))
    except Exception:
        pass
    return None

def fold_key_from_path(path):
    bn = os.path.basename(path)
    for tok in bn.replace(".pt","").replace("-","_").split("_"):
        if tok.lower().startswith("fold"):
            try: return int(tok.lower().replace("fold",""))
            except: pass
    m = re.findall(r"(\d+)", bn)
    return int(m[-1]) if m else -1

# ---------- Safe loading ----------
def detect_gatv2_heads_from_sd(sd: dict) -> int:
    for k, v in sd.items():
        if isinstance(v, torch.Tensor) and k.endswith("att") and v.dim() == 3:
            return int(v.shape[1])
    return 1

def build_model_for_sd(sd: dict, dims):
    X_DIM, G_DIM, R_DIM, E_DIM = dims["x"], dims["g"], dims["r"], dims["edge"]
    keys = list(sd.keys())
    keys_join = "_".join(keys).lower()
    if any(("g1." in k and "gat" in keys_join) for k in keys):
        heads = detect_gatv2_heads_from_sd(sd)
        return GATv2_DX_DG_DR_H4(X_DIM, G_DIM, R_DIM, E_DIM) if heads >= 4 else GATv2_DX_DG_DR(X_DIM, G_DIM, R_DIM, E_DIM)
    if any(("c1." in k and "lin_edge" in k) for k in keys):
        return Transformer_DX_DG_DR(X_DIM, G_DIM, R_DIM, E_DIM)
    if any(("c1.weight" in k or "n1." in k) for k in keys):
        return NNConv_DX_DG_DR(X_DIM, G_DIM, R_DIM, E_DIM)
    return Transformer_DX_DG_DR(X_DIM, G_DIM, R_DIM, E_DIM)

def safe_load_compatible_tensors(model: nn.Module, sd: dict):
    msd = model.state_dict()
    sd_f = {}
    for k, v in sd.items():
        if k in msd and isinstance(v, torch.Tensor) and msd[k].shape == v.shape:
            sd_f[k] = v
    if len(sd_f) < len(msd):
        log(f"[WARN] load compatible tensors {len(sd_f)}/{len(msd)}; skip {len(msd)-len(sd_f)} keys")
    model.load_state_dict(sd_f, strict=False)
    return model

@torch.no_grad()
def predict_family_probs(fid, fam_defs_tbl, dims, data_list):
    fam = fam_defs_tbl[fid]
    files = sorted(set(fam["gatv2"] + fam["trans"] + fam["nn"]))
    if not files:
        raise RuntimeError(f"No weight files for {fid} in {fam['dir']}")
    fmap = {}
    for p in files:
        f = fold_key_from_path(p)
        fmap.setdefault(f, []).append(p)

    dl = DataLoader(data_list, batch_size=BATCH, shuffle=False)
    fold_probs = []
    for f, plist in sorted(fmap.items()):
        probs_fold = []
        for wt in plist:
            sd = torch.load(wt, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            model = build_model_for_sd(sd, dims).to(DEVICE)
            model = safe_load_compatible_tensors(model, sd).eval()
            ps = []
            for bt in dl:
                bt = bt.to(DEVICE)
                ps.append(torch.sigmoid(model(bt)).detach().cpu())
            probs_fold.append(torch.cat(ps).numpy())
            del model
        if probs_fold:
            fold_probs.append(np.mean(np.stack(probs_fold, axis=0), axis=0))

    fam_probs = np.mean(np.stack(fold_probs, axis=0), axis=0)
    fam_weight = infer_weights_from_val_csv(fam["val_csvs"]) or 1.0
    return fam_probs, fam_weight

# ---------- Metrics ----------
def roc_auc(y, p):
    try: return float(roc_auc_score(y, p))
    except Exception: return float("nan")

def pr_auc(y, p):
    try:
        prec, rec, _ = precision_recall_curve(y, p)
        return float(auc(rec, prec))
    except Exception:
        return float("nan")

def f1_at(y, p, thr):
    try: return float(f1_score(y, (p >= thr).astype(int)))
    except Exception: return float("nan")

def best_f1(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    thr = np.append(thr, 1.0)
    f1  = 2*prec*rec/np.clip(prec+rec, 1e-9, None)
    i   = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])

def precision_at_k(y, p, k):
    if len(p) == 0: return float("nan")
    idx = np.argsort(-p)[:min(k, len(p))]
    if len(idx) == 0: return float("nan")
    return float(np.mean(np.array(y)[idx]))

def metrics_block(y_true, p_pred):
    out = {
        "ROC": roc_auc(y_true, p_pred),
        "PR":  pr_auc(y_true, p_pred),
        "F1@0.55": f1_at(y_true, p_pred, FIXED_DECISION_THRESHOLD)
    }
    bt, bf1 = best_f1(y_true, p_pred)
    out["BestThr"] = bt
    out["F1@Best"] = bf1
    return out

def metrics_with_strata(prefix, y, p, tiers, p_at_k_list):
    rows = {}
    m_all = metrics_block(y, p)
    rows.update({f"{prefix}_ROC":m_all["ROC"], f"{prefix}_PR":m_all["PR"],
                 f"{prefix}_F1@0.55":m_all["F1@0.55"], f"{prefix}_BestThr":m_all["BestThr"],
                 f"{prefix}_F1@Best":m_all["F1@Best"]})
    tiers = np.array([str(t).lower() for t in tiers], dtype=object)
    mask_s = np.array([t.startswith("strong_") for t in tiers], dtype=bool)
    mask_w = (tiers == "weak_pos")
    if mask_s.any():
        m_s = metrics_block(y[mask_s], p[mask_s])
        rows.update({f"{prefix}_Strong_ROC":m_s["ROC"], f"{prefix}_Strong_PR":m_s["PR"],
                     f"{prefix}_Strong_F1@0.55":m_s["F1@0.55"], f"{prefix}_Strong_BestThr":m_s["BestThr"],
                     f"{prefix}_Strong_F1@Best":m_s["F1@Best"]})
    else:
        rows.update({f"{prefix}_Strong_ROC":np.nan, f"{prefix}_Strong_PR":np.nan,
                     f"{prefix}_Strong_F1@0.55":np.nan, f"{prefix}_Strong_BestThr":np.nan,
                     f"{prefix}_Strong_F1@Best":np.nan})
    if mask_w.any():
        m_w = metrics_block(y[mask_w], p[mask_w])
        rows.update({f"{prefix}_Weak_ROC":m_w["ROC"], f"{prefix}_Weak_PR":m_w["PR"],
                     f"{prefix}_Weak_F1@0.55":m_w["F1@0.55"], f"{prefix}_Weak_BestThr":m_w["BestThr"],
                     f"{prefix}_Weak_F1@Best":m_w["F1@Best"]})
        for K in p_at_k_list:
            rows[f"{prefix}_Weak_P@{K}"] = precision_at_k(y[mask_w], p[mask_w], K)
    else:
        rows.update({f"{prefix}_Weak_ROC":np.nan, f"{prefix}_Weak_PR":np.nan,
                     f"{prefix}_Weak_F1@0.55":np.nan, f"{prefix}_Weak_BestThr":np.nan,
                     f"{prefix}_Weak_F1@Best":np.nan})
        for K in p_at_k_list:
            rows[f"{prefix}_Weak_P@{K}"] = np.nan
    return rows

# ---------- Extractors ----------
def valid_mask(data_list):
    ys = np.array([int(getattr(d, "y", -1)) for d in data_list])
    return (ys == 0) | (ys == 1)

def extract_labels(data_list, mask):
    return np.array([int(getattr(d, "y", -1)) for i, d in enumerate(data_list) if mask[i]], dtype=int)

def extract_tiers(data_list, mask):
    return np.array([str(getattr(d, "tier", "")).lower() for i, d in enumerate(data_list) if mask[i]], dtype=object)

# ---------- Combine probs ----------
def combine_probs(combo, table, mode, fam_weights):
    probs = [table[f] for f in combo]
    if mode == "equal":
        w = np.ones(len(combo)) / len(combo)
    else:
        ws = np.array([max(1e-9, fam_weights[f]) for f in combo], dtype=float)
        w = ws / ws.sum()
    return np.sum(np.stack([w[i]*probs[i] for i in range(len(combo))], axis=0), axis=0), w

# ---------- Strict external builder ----------
def build_strict_external():
    log("== Build strict external dataset ==")
    if not Path(PT_EXTERNAL).exists():
        raise FileNotFoundError(f"Missing external PT: {PT_EXTERNAL}")
    if not Path(IDX_EXTERNAL).exists():
        raise FileNotFoundError(f"Missing external graph index CSV: {IDX_EXTERNAL}")

    # internal sources
    idx_internal_df = choose_metadata_table(read_csv_flex(IDX_INTERNAL), "internal_graph_index") if Path(IDX_INTERNAL).exists() else None
    curated_internal_df = choose_metadata_table(read_csv_flex(INTERNAL_CURATED_CSV), "internal_curated") if Path(INTERNAL_CURATED_CSV).exists() else None

    if idx_internal_df is None and curated_internal_df is None:
        raise FileNotFoundError("At least one internal source is required: data_graph_with_smiles_index.csv or merged_for_model_consolidated.csv")

    # external graph index
    ext_df = choose_metadata_table(read_csv_flex(IDX_EXTERNAL), "external_graph_index")
    ext_raw_df = choose_metadata_table(read_csv_flex(EXTERNAL_COMPOUND_CSV), "external_compound") if Path(EXTERNAL_COMPOUND_CSV).exists() else None

    # internal overlap sets
    cid_set = set()
    can_set = set()
    for df in [idx_internal_df, curated_internal_df]:
        if df is None: 
            continue
        cid_set |= set(df["cid_norm"].dropna().astype(int).tolist())
        can_set |= set(df["canonical_smiles"].dropna().astype(str).tolist())

    # annotate external graph index
    ext_df = ext_df.copy().reset_index(drop=True)
    ext_df["graph_row_id"] = np.arange(len(ext_df))
    ext_df["overlap_cid_internal"] = ext_df["cid_norm"].map(lambda x: (int(x) in cid_set) if pd.notna(x) else False)
    ext_df["overlap_canonical_internal"] = ext_df["canonical_smiles"].map(lambda x: (str(x) in can_set) if pd.notna(x) else False)

    # optionally harmonize from raw external compound CSV (same CID/canonical info could help QC only)
    if ext_raw_df is not None:
        ext_df["present_in_external_compound_csv_by_cid"] = ext_df["cid_norm"].map(
            lambda x: ext_raw_df["cid_norm"].eq(x).any() if pd.notna(x) else False
        )
    else:
        ext_df["present_in_external_compound_csv_by_cid"] = False

    if STRICT_OVERLAP_RULE == "cid_only":
        ext_df["keep_strict"] = ~ext_df["overlap_cid_internal"]
    elif STRICT_OVERLAP_RULE == "canonical_only":
        ext_df["keep_strict"] = ~ext_df["overlap_canonical_internal"]
    else:
        ext_df["keep_strict"] = ~(ext_df["overlap_cid_internal"] | ext_df["overlap_canonical_internal"])

    ext_df["murcko_scaffold"] = ext_df["canonical_smiles"].map(murcko_smiles)

    ext_df.to_csv(STRICT_EXTERNAL_ANNOT, index=False)
    strict_df = ext_df.loc[ext_df["keep_strict"]].copy().reset_index(drop=True)
    strict_df.to_csv(STRICT_EXTERNAL_CSV, index=False)

    # create strict .pt by row filtering
    data_all = load_pyg_list(PT_EXTERNAL)
    if len(data_all) != len(ext_df):
        raise RuntimeError(
            f"Row mismatch: len(data_graph_external.pt)={len(data_all)} vs len(data_graph_external_index.csv)={len(ext_df)}. "
            "strict PT can only be built when row order matches."
        )

    keep_mask = strict_df["graph_row_id"].tolist()
    strict_data = [data_all[i] for i in keep_mask]
    if SAVE_STRICT_PT:
        torch.save(strict_data, STRICT_EXTERNAL_PT)

    # summary
    summary = {
        "n_external_graph_rows": int(len(ext_df)),
        "n_strict_external_graph_rows": int(len(strict_df)),
        "rule": STRICT_OVERLAP_RULE,
        "n_overlap_cid": int(ext_df["overlap_cid_internal"].sum()),
        "n_overlap_canonical": int(ext_df["overlap_canonical_internal"].sum()),
        "n_removed_total": int((~ext_df["keep_strict"]).sum()),
        "strict_pt_path": STRICT_EXTERNAL_PT,
    }
    Path(f"{ROOT}/strict_external_build_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(json.dumps(summary, indent=2, ensure_ascii=False))
    return strict_data, strict_df

# ---------- Plot helpers ----------
def _new_curve_figure():
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    fig.subplots_adjust(left=PLOT_LEFT, right=PLOT_RIGHT, bottom=PLOT_BOTTOM, top=PLOT_TOP)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=TICK_LABELSIZE, width=SPINE_LINEWIDTH, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_LINEWIDTH)
    return fig, ax

def save_roc_curve(y, p, out_png, out_points_csv, title=None):
    fpr, tpr, thr = roc_curve(y, p)
    aucv = roc_auc(y, p)
    pts = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr})
    pts.to_csv(out_points_csv, index=False)

    fig, ax = _new_curve_figure()
    ax.plot(fpr, tpr, color="tab:blue", linewidth=CURVE_LINEWIDTH, label=f"AUC={aucv:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="tab:orange", linewidth=DIAG_LINEWIDTH)
    ax.set_xlabel("False Positive Rate", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=AXIS_LABELSIZE)
    ax.legend(loc=PLOT_ROC_LEGEND_LOC, frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)

def save_pr_curve(y, p, out_png, out_points_csv, title=None):
    precision, recall, thr = precision_recall_curve(y, p)
    aucv = auc(recall, precision)
    thr = np.append(thr, 1.0)
    pts = pd.DataFrame({"recall": recall, "precision": precision, "threshold": thr})
    pts.to_csv(out_points_csv, index=False)

    fig, ax = _new_curve_figure()
    ax.plot(recall, precision, color="tab:blue", linewidth=CURVE_LINEWIDTH, label=f"AUC={aucv:.4f}")
    ax.set_xlabel("Recall", fontsize=AXIS_LABELSIZE)
    ax.set_ylabel("Precision", fontsize=AXIS_LABELSIZE)
    ax.legend(loc=PLOT_PR_LEGEND_LOC, frameon=True, fontsize=LEGEND_FONTSIZE)
    fig.savefig(out_png, dpi=PLOT_DPI)
    plt.close(fig)

def annotate_confusion(df_meta, y, p, thr, label):
    pred = (p >= thr).astype(int)
    out = df_meta.copy()
    out[f"pred_label_{label}"] = pred
    out[f"threshold_{label}"] = thr
    conds = []
    for yt, yp in zip(y, pred):
        if yt == 1 and yp == 1:
            conds.append("TP")
        elif yt == 0 and yp == 0:
            conds.append("TN")
        elif yt == 0 and yp == 1:
            conds.append("FP")
        else:
            conds.append("FN")
    out[f"confusion_{label}"] = conds
    return out

def confusion_counts(y, p, thr, label):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return pd.DataFrame([{
        "threshold_label": label,
        "threshold": float(thr),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn)
    }])

# ---------- Main ----------
def main():
    # 1) strict external dataset construction
    strict_data_all, strict_meta = build_strict_external()

    # 2) valid labels only
    m_valid = valid_mask(strict_data_all)
    data = [d for i, d in enumerate(strict_data_all) if m_valid[i]]
    strict_meta_valid = strict_meta.loc[m_valid].reset_index(drop=True).copy()
    if len(data) == 0:
        raise RuntimeError("strict external dataset に有効な y(0/1) がありません。")

    dims = infer_dims(data)
    y = extract_labels(data, np.ones(len(data), dtype=bool))
    tiers = extract_tiers(data, np.ones(len(data), dtype=bool))
    strict_meta_valid["y_true"] = y
    strict_meta_valid["tier_from_pt"] = tiers

    log(f"[INFO] strict external n_valid = {len(data)}")
    log(f"[INFO] dims = {dims}")

    # 3) family prediction
    fam_defs_tbl = family_defs()
    log("🔎 Family directories (auto-detected):")
    for k, v in fam_defs_tbl.items():
        log(f"  {k}: {v['dir']}")

    fam_ids = ["M1", "M2", "M3", "M4"]
    fam_probs_tbl = {}
    fam_weights = {}
    for fid in fam_ids:
        log(f"== Predict family {fid} ==")
        pu, w = predict_family_probs(fid, fam_defs_tbl, dims, data)
        fam_probs_tbl[fid] = pu
        fam_weights[fid] = w
        log(f"   weight(cv_pr hint) = {w}")

    # 4) compare all combos
    singles = [(f,) for f in fam_ids]
    pairs   = list(itertools.combinations(fam_ids, 2))
    trips   = list(itertools.combinations(fam_ids, 3))
    quads   = [tuple(fam_ids)]
    combos  = singles + pairs + trips + quads

    rows = []
    prob_cache = {}
    for mode in ["equal", "cv_pr"]:
        for combo in combos:
            name = "+".join(combo)
            p_u, w = combine_probs(combo, fam_probs_tbl, mode, fam_weights)
            prob_cache[(mode, name)] = p_u
            rec = {"Mode": mode, "Combo": name, "Weights": json.dumps({combo[i]: float(w[i]) for i in range(len(combo))}, ensure_ascii=False)}
            rec.update(metrics_with_strata("StrictExternal", y, p_u, tiers, P_AT_K))
            rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_ALL, index=False)

    # ranking by PR then ROC
    df_rank = df.copy()
    df_rank["RankScore"] = -df_rank["StrictExternal_PR"].fillna(-1e9) - 1e-3*df_rank["StrictExternal_ROC"].fillna(-1e9)
    df_rank = df_rank.sort_values(["RankScore"], ascending=True).reset_index(drop=True)
    df_rank.to_csv(OUT_RANK, index=False)

    log("\n=== Top 10 by StrictExternal_PR then ROC ===")
    cols = ["Mode","Combo","StrictExternal_PR","StrictExternal_ROC","StrictExternal_Weak_PR","StrictExternal_Strong_PR"]
    log(df_rank[cols].head(10).to_string(index=False))

    # 5) best combo details
    best = df_rank.iloc[0].to_dict()
    best_mode = best["Mode"]
    best_combo = best["Combo"]
    p_best = prob_cache[(best_mode, best_combo)]

    best_metrics = metrics_block(y, p_best)
    best_thr = float(best_metrics["BestThr"])
    strict_meta_valid = strict_meta_valid.copy()
    strict_meta_valid["pred_prob"] = p_best
    strict_meta_valid["best_mode"] = best_mode
    strict_meta_valid["best_combo"] = best_combo

    # confusion annotations (fixed internal decision threshold = 0.55)
    out_best = annotate_confusion(strict_meta_valid, y, p_best, FIXED_DECISION_THRESHOLD, "thr0.55")
    out_best.to_csv(OUT_BESTP, index=False)

    conf_df = confusion_counts(y, p_best, FIXED_DECISION_THRESHOLD, "thr0.55")
    conf_df.to_csv(OUT_CONF, index=False)

    # ROC / PR plots
    save_roc_curve(y, p_best, OUT_ROC, OUT_ROCPT, f"Strict external ROC ({best_mode} | {best_combo})")
    save_pr_curve(y, p_best, OUT_PR, OUT_PRPT, f"Strict external PR ({best_mode} | {best_combo})")

    # metrics JSON
    metrics_json = {
        "best_mode": best_mode,
        "best_combo": best_combo,
        "n_strict_external_valid": int(len(y)),
        "StrictExternal_ROC": float(best_metrics["ROC"]),
        "StrictExternal_PR": float(best_metrics["PR"]),
        "StrictExternal_F1_at_0.55": float(best_metrics["F1@0.55"]),
        "StrictExternal_fixed_decision_threshold": float(FIXED_DECISION_THRESHOLD),
        "StrictExternal_best_threshold": float(best_metrics["BestThr"]),
        "StrictExternal_F1_at_best_threshold": float(best_metrics["F1@Best"]),
        "strict_rule": STRICT_OVERLAP_RULE,
        "output_files": {
            "comparison_all_csv": OUT_ALL,
            "ranking_csv": OUT_RANK,
            "best_predictions_csv": OUT_BESTP,
            "best_confusion_counts_csv": OUT_CONF,
            "best_roc_png": OUT_ROC,
            "best_pr_png": OUT_PR,
            "best_roc_points_csv": OUT_ROCPT,
            "best_pr_points_csv": OUT_PRPT,
            "strict_overlap_annotation_csv": STRICT_EXTERNAL_ANNOT,
            "strict_external_csv": STRICT_EXTERNAL_CSV,
            "strict_external_pt": STRICT_EXTERNAL_PT,
        }
    }
    Path(OUT_METR).write_text(json.dumps(metrics_json, indent=2, ensure_ascii=False))

    log("\n✅ Done.")
    log(f"📁 {OUT_ALL}")
    log(f"🏁 {OUT_RANK}")
    log(f"📄 {OUT_BESTP}")
    log(f"📄 {OUT_CONF}")
    log(f"🖼️ {OUT_ROC}")
    log(f"🖼️ {OUT_PR}")
    log(json.dumps(metrics_json, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
