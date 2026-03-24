#!/usr/bin/env python3
"""
Comment 5 (training/test assembly and similarity) analysis script for Chemoinfo_QT.

What this script does
---------------------
1) Builds explicit split-membership tables for:
   - M2: scaffold 5-fold CV (graph-level, matching the notebook logic)
   - M4: Butina cluster leave-cluster-out 5-fold CV (graph-level, matching the notebook logic)
2) Quantifies train-test chemical relatedness using nearest-neighbor Tanimoto similarity.
3) Audits internal-vs-external overlap by CID / raw SMILES / canonical SMILES.
4) Creates a stricter compound-disjoint external set.
5) Filters existing external prediction outputs to the strict external subset and recomputes metrics.
6) Produces manuscript-ready CSV / JSON / PNG outputs and a short markdown summary.

Notes
-----
- This script does NOT retrain or re-infer models. For the strict external set, it filters the
  already generated external prediction CSV to compounds retained after strict overlap removal.
- The script is intentionally conservative and explicit, so the output files can be cited in the
  manuscript / supplementary information / response letter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score, roc_curve

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
RDLogger.DisableLog("rdApp.*")


@dataclass
class Paths:
    root: Path
    internal_compound_csv: Path
    internal_graph_index_csv: Path
    external_compound_csv: Path
    external_graph_index_csv: Path
    external_predictions_csv: Optional[Path]
    output_dir: Path
    external_graph_pt: Optional[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comment 5 split/similarity analysis")
    parser.add_argument("--root", type=str, default=".", help="Chemoinfo_QT project root")
    parser.add_argument("--output_dir", type=str, default="comment5_analysis", help="Output directory")
    parser.add_argument("--internal_compound_csv", type=str, default="merged_for_model_consolidated.csv")
    parser.add_argument("--internal_graph_index_csv", type=str, default="data_graph_with_smiles_index.csv")
    parser.add_argument("--external_compound_csv", type=str, default="external_AID_588834_herg.csv")
    parser.add_argument("--external_graph_index_csv", type=str, default="data_graph_external_index.csv")
    parser.add_argument(
        "--external_predictions_csv",
        type=str,
        default="reports_external_confusions_manual/predictions_external_M2M4_cvpr.csv",
        help="Existing external prediction CSV for strict-subset metric recalculation",
    )
    parser.add_argument(
        "--external_graph_pt",
        type=str,
        default="data_graph_external.pt",
        help="Optional graph .pt file to subset for the strict external set",
    )
    parser.add_argument("--morgan_radius", type=int, default=2)
    parser.add_argument("--morgan_nbits", type=int, default=2048)
    parser.add_argument("--butina_threshold", type=float, default=0.7)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--min_pos_per_fold", type=int, default=10)
    parser.add_argument("--min_neg_per_fold", type=int, default=10)
    parser.add_argument("--fold_assign_trials", type=int, default=200)
    parser.add_argument(
        "--strict_overlap_rule",
        type=str,
        default="cid_or_canonical",
        choices=["cid_or_canonical", "cid_only", "canonical_only"],
        help="Rule used to define strict external compounds excluded by internal overlap",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> Paths:
    root = Path(args.root).resolve()
    out = (root / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    def maybe(path_str: str) -> Optional[Path]:
        p = (root / path_str).resolve()
        return p if p.exists() else None

    paths = Paths(
        root=root,
        internal_compound_csv=(root / args.internal_compound_csv).resolve(),
        internal_graph_index_csv=(root / args.internal_graph_index_csv).resolve(),
        external_compound_csv=(root / args.external_compound_csv).resolve(),
        external_graph_index_csv=(root / args.external_graph_index_csv).resolve(),
        external_predictions_csv=maybe(args.external_predictions_csv),
        output_dir=out,
        external_graph_pt=maybe(args.external_graph_pt),
    )

    required = [
        paths.internal_compound_csv,
        paths.internal_graph_index_csv,
        paths.external_compound_csv,
        paths.external_graph_index_csv,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required files not found: {missing}")
    return paths


def choose_col(df: pd.DataFrame, candidates: Sequence[str], required: bool = True) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Could not find any of columns: {candidates}; available={list(df.columns)}")
    return None


def canonicalize_smiles(s: object) -> Optional[str]:
    if s is None or pd.isna(s):
        return None
    smi = str(s).strip()
    if not smi or smi.lower() in {"nan", "none"}:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def murcko_scaffold_from_smiles(s: object) -> Optional[str]:
    can = canonicalize_smiles(s)
    if can is None:
        return None
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return None
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return None


def morgan_fp_from_smiles(s: object, radius: int, nbits: int):
    can = canonicalize_smiles(s)
    if can is None:
        return None
    mol = Chem.MolFromSmiles(can)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def add_standard_columns(
    df: pd.DataFrame,
    smiles_candidates: Sequence[str],
    cid_candidates: Sequence[str],
    label_candidates: Sequence[str],
    dataset_name: str,
) -> pd.DataFrame:
    out = df.copy()
    smi_col = choose_col(out, smiles_candidates)
    cid_col = choose_col(out, cid_candidates, required=False)
    lab_col = choose_col(out, label_candidates, required=False)

    out["SMILES_raw"] = out[smi_col].astype(str)
    out["SMILES_canonical"] = out[smi_col].map(canonicalize_smiles)
    if cid_col is not None:
        out["CID_std"] = pd.to_numeric(out[cid_col], errors="coerce").astype("Int64")
    else:
        out["CID_std"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
    if lab_col is not None:
        out["label_std"] = pd.to_numeric(out[lab_col], errors="coerce").astype("Int64")
    else:
        out["label_std"] = pd.Series(pd.array([pd.NA] * len(out), dtype="Int64"))
    out["dataset_name"] = dataset_name
    return out


def nearest_neighbor_similarity(train_fps: Sequence, test_fps: Sequence) -> np.ndarray:
    scores = np.full(len(test_fps), np.nan, dtype=float)
    valid_train = [fp for fp in train_fps if fp is not None]
    if not valid_train:
        return scores
    for i, fp in enumerate(test_fps):
        if fp is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp, valid_train)
        scores[i] = float(max(sims)) if sims else np.nan
    return scores


def summarize_similarity(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "q1": np.nan,
            "q3": np.nan,
            "min": np.nan,
            "max": np.nan,
            "exact_1.0_n": 0,
            "exact_1.0_pct": np.nan,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.quantile(arr, 0.25)),
        "q3": float(np.quantile(arr, 0.75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "exact_1.0_n": int(np.sum(np.isclose(arr, 1.0))),
        "exact_1.0_pct": float(np.mean(np.isclose(arr, 1.0)) * 100.0),
    }


def save_boxplot(groups: Dict[str, Sequence[float]], out_png: Path, title: str, ylabel: str = "Nearest-neighbor Tanimoto") -> None:
    labels = []
    data = []
    for label, values in groups.items():
        arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
        if arr.size == 0:
            continue
        labels.append(label)
        data.append(arr)
    if not data:
        return
    plt.figure(figsize=(max(8, len(labels) * 0.8), 5.5))
    plt.boxplot(data, tick_labels=labels, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def save_histogram(values: Sequence[float], out_png: Path, title: str, xlabel: str = "Nearest-neighbor Tanimoto") -> None:
    arr = np.asarray([v for v in values if pd.notna(v)], dtype=float)
    if arr.size == 0:
        return
    plt.figure(figsize=(6.5, 4.8))
    plt.hist(arr, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


def summary_row(name: str, df: pd.DataFrame) -> Dict[str, object]:
    return {
        "dataset": name,
        "n_rows": int(len(df)),
        "n_unique_cid": int(df["CID_std"].dropna().nunique()),
        "n_unique_raw_smiles": int(df["SMILES_raw"].dropna().nunique()),
        "n_unique_canonical_smiles": int(df["SMILES_canonical"].dropna().nunique()),
        "n_label_pos": int((df["label_std"] == 1).sum()) if "label_std" in df.columns else np.nan,
        "n_label_neg": int((df["label_std"] == 0).sum()) if "label_std" in df.columns else np.nan,
    }


def build_m2_scaffold_folds(graph_df: pd.DataFrame, n_folds: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    scaf = []
    for i, smi in enumerate(graph_df["SMILES_raw"]):
        key = murcko_scaffold_from_smiles(smi)
        if key is None:
            key = f"_NOSCAF_{i}"
        scaf.append(key)
    graph_df = graph_df.copy()
    graph_df["m2_scaffold"] = scaf

    scaf_to_indices: Dict[str, List[int]] = {}
    for idx, s in enumerate(scaf):
        scaf_to_indices.setdefault(s, []).append(idx)

    scaffold_keys = list(scaf_to_indices.keys())
    random.Random(SEED).shuffle(scaffold_keys)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for i, s in enumerate(scaffold_keys):
        folds[i % n_folds].extend(scaf_to_indices[s])

    fold_of = {}
    for fold, idxs in enumerate(folds):
        for idx in idxs:
            fold_of[idx] = fold

    graph_df["m2_fold"] = graph_df.index.map(fold_of)
    graph_df["m2_role"] = "test"
    graph_df["m2_split_method"] = "Murcko scaffold 5-fold CV"

    fold_summary = (
        graph_df.groupby("m2_fold", dropna=False)
        .agg(
            n_rows=("m2_fold", "size"),
            n_pos=("label_std", lambda s: int((s == 1).sum())),
            n_neg=("label_std", lambda s: int((s == 0).sum())),
            n_scaffolds=("m2_scaffold", "nunique"),
            n_unique_cid=("CID_std", pd.Series.nunique),
            n_unique_canonical_smiles=("SMILES_canonical", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"m2_fold": "fold"})
    )
    return graph_df, fold_summary


def cluster_butina(fps: Sequence, threshold: float) -> List[List[int]]:
    dists = []
    for i in range(1, len(fps)):
        fp_i = fps[i]
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, fps[:i]) if fp_i is not None else [0.0] * i
        dists.extend([1.0 - float(s) for s in sims])
    clusters = Butina.ClusterData(dists, len(fps), 1.0 - float(threshold), isDistData=True)
    return [list(c) for c in clusters]


def summarize_cluster_labels(labels: Sequence[int], clusters: Sequence[Sequence[int]]) -> pd.DataFrame:
    rows = []
    labels = list(labels)
    for cid, idxs in enumerate(clusters):
        ys = [int(labels[i]) for i in idxs]
        rows.append({"cluster_id": cid, "size": len(idxs), "pos": sum(ys), "neg": len(ys) - sum(ys)})
    return pd.DataFrame(rows)


def try_pack_clusters(
    cluster_label_df: pd.DataFrame,
    k: int,
    min_pos: int,
    min_neg: int,
    trials: int,
) -> Tuple[List[Dict[str, object]], Tuple[int, float]]:
    sizes = cluster_label_df.set_index("cluster_id")["size"].to_dict()
    pos_c = cluster_label_df.set_index("cluster_id")["pos"].to_dict()
    neg_c = cluster_label_df.set_index("cluster_id")["neg"].to_dict()
    cluster_ids = cluster_label_df.sort_values("size", ascending=False)["cluster_id"].tolist()

    best_solution = None
    best_score = (10**9, float("inf"))

    for _ in range(trials):
        folds = [{"cluster_ids": [], "pos": 0, "neg": 0, "size": 0} for _ in range(k)]
        for cid in cluster_ids:
            best_fold = None
            best_tuple = (10**9, float("inf"))
            for f in range(k):
                p = folds[f]["pos"] + pos_c[cid]
                n = folds[f]["neg"] + neg_c[cid]
                s = folds[f]["size"] + sizes[cid]
                viol = int(p < min_pos) + int(n < min_neg)
                score = (viol, s)
                if score < best_tuple:
                    best_tuple = score
                    best_fold = f
            folds[best_fold]["cluster_ids"].append(cid)
            folds[best_fold]["pos"] += pos_c[cid]
            folds[best_fold]["neg"] += neg_c[cid]
            folds[best_fold]["size"] += sizes[cid]

        viol = sum(int(fd["pos"] < min_pos or fd["neg"] < min_neg) for fd in folds)
        size_var = float(np.var([fd["size"] for fd in folds]))
        score = (viol, size_var)
        if score < best_score:
            best_score = score
            best_solution = folds
            if viol == 0:
                break
    return best_solution, best_score


def assign_folds_with_relaxation(
    labels: Sequence[int],
    clusters: Sequence[Sequence[int]],
    k: int,
    min_pos: int,
    min_neg: int,
    trials: int,
) -> Tuple[Dict[int, int], pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    cluster_label_df = summarize_cluster_labels(labels, clusters)
    folds, score = try_pack_clusters(cluster_label_df, k=k, min_pos=min_pos, min_neg=min_neg, trials=trials)
    viol, size_var = score
    relax_steps = 0
    mpos, mneg = min_pos, min_neg

    while viol > 0 and (mpos > 0 or mneg > 0):
        relax_steps += 1
        if mpos > 0:
            mpos -= 1
        if mneg > 0:
            mneg -= 1
        folds2, score2 = try_pack_clusters(cluster_label_df, k=k, min_pos=mpos, min_neg=mneg, trials=trials)
        if score2 < score:
            folds, score = folds2, score2
            viol, size_var = score
        if viol == 0:
            break

    node2fold = {}
    for fold, info in enumerate(folds):
        for cluster_id in info["cluster_ids"]:
            for idx in clusters[cluster_id]:
                node2fold[idx] = fold

    fold_summary = pd.DataFrame(
        [{"fold": f, "n_rows": fd["size"], "n_pos": fd["pos"], "n_neg": fd["neg"]} for f, fd in enumerate(folds)]
    )

    cluster_label_df = cluster_label_df.copy()
    cluster_label_df["fold"] = -1
    for fold, info in enumerate(folds):
        for cluster_id in info["cluster_ids"]:
            cluster_label_df.loc[cluster_label_df["cluster_id"] == cluster_id, "fold"] = fold

    extra = {
        "violations": int(viol),
        "size_var": float(size_var),
        "relax_steps": int(relax_steps),
        "min_pos_final": int(mpos),
        "min_neg_final": int(mneg),
    }
    return node2fold, fold_summary, cluster_label_df, extra


def build_m4_lco_folds(
    graph_df: pd.DataFrame,
    radius: int,
    nbits: int,
    butina_threshold: float,
    n_folds: int,
    min_pos_per_fold: int,
    min_neg_per_fold: int,
    fold_assign_trials: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    graph_df = graph_df.copy()

    # To keep the analysis tractable, clustering is done at the unique canonical-SMILES level,
    # then propagated back to augmented graph rows. This preserves the chemical-space split logic
    # while avoiding repeated clustering of duplicated augmented entries.
    unique_df = (
        graph_df.loc[:, ["SMILES_canonical", "label_std"]]
        .dropna(subset=["SMILES_canonical"])
        .drop_duplicates(subset=["SMILES_canonical"])
        .reset_index(drop=True)
    )
    fps = [morgan_fp_from_smiles(s, radius=radius, nbits=nbits) for s in unique_df["SMILES_canonical"]]
    zero_mol = Chem.MolFromSmiles("CC")
    zero_fp = AllChem.GetMorganFingerprintAsBitVect(zero_mol, radius, nBits=nbits)
    fps_safe = [fp if fp is not None else zero_fp for fp in fps]

    clusters = cluster_butina(fps_safe, threshold=butina_threshold)
    while len(clusters) < n_folds:
        clusters = sorted(clusters, key=len)
        a = clusters.pop(0)
        b = clusters.pop(0)
        clusters.append(a + b)

    node2fold, fold_summary_unique, cluster_label_df, extra = assign_folds_with_relaxation(
        labels=unique_df["label_std"].fillna(0).astype(int).tolist(),
        clusters=clusters,
        k=n_folds,
        min_pos=min_pos_per_fold,
        min_neg=min_neg_per_fold,
        trials=fold_assign_trials,
    )

    idx_to_cluster = {}
    for cluster_id, idxs in enumerate(clusters):
        for idx in idxs:
            idx_to_cluster[idx] = cluster_id
    unique_df["m4_fold"] = unique_df.index.map(node2fold)
    unique_df["m4_cluster_id"] = unique_df.index.map(idx_to_cluster)

    fold_map = unique_df.set_index("SMILES_canonical")["m4_fold"].to_dict()
    cluster_map = unique_df.set_index("SMILES_canonical")["m4_cluster_id"].to_dict()
    graph_df["m4_fold"] = graph_df["SMILES_canonical"].map(fold_map)
    graph_df["m4_cluster_id"] = graph_df["SMILES_canonical"].map(cluster_map)
    graph_df["m4_role"] = "test"
    graph_df["m4_split_method"] = f"Butina cluster LCO 5-fold (Tanimoto >= {butina_threshold})"

    fold_summary = (
        graph_df.groupby("m4_fold", dropna=False)
        .agg(
            n_rows=("m4_fold", "size"),
            n_pos=("label_std", lambda s: int((s == 1).sum())),
            n_neg=("label_std", lambda s: int((s == 0).sum())),
            n_clusters=("m4_cluster_id", "nunique"),
            n_unique_cid=("CID_std", pd.Series.nunique),
            n_unique_canonical_smiles=("SMILES_canonical", pd.Series.nunique),
        )
        .reset_index()
        .rename(columns={"m4_fold": "fold"})
    )
    extra["n_unique_canonical_for_clustering"] = int(len(unique_df))
    fold_summary_unique = fold_summary_unique.rename(columns={"n_rows": "n_unique_rows"})
    return graph_df, fold_summary, cluster_label_df, extra


def compute_fold_similarity(
    df: pd.DataFrame,
    fold_col: str,
    split_name: str,
    radius: int,
    nbits: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    df = df.copy()
    # Heavy graph-level duplication exists because of augmentation.
    # To keep runtime practical while preserving the chemical-space comparison,
    # similarity is computed on unique canonical SMILES within each train/test partition,
    # then propagated back to graph rows.
    unique_can = df["SMILES_canonical"].fillna("_NA_")
    fp_cache = {}
    for can in pd.unique(unique_can):
        if can == "_NA_":
            fp_cache[can] = None
        else:
            fp_cache[can] = morgan_fp_from_smiles(can, radius=radius, nbits=nbits)

    per_row = []
    per_fold = []
    fold_distributions: Dict[str, np.ndarray] = {}
    folds = sorted([f for f in df[fold_col].dropna().unique()])
    for fold in folds:
        test_df = df.loc[df[fold_col] == fold].copy()
        train_df = df.loc[df[fold_col] != fold].copy()

        train_unique = pd.unique(train_df["SMILES_canonical"].fillna("_NA_"))
        test_unique = pd.unique(test_df["SMILES_canonical"].fillna("_NA_"))

        train_fps = [fp_cache[c] for c in train_unique]
        test_fps = [fp_cache[c] for c in test_unique]
        unique_sims = nearest_neighbor_similarity(train_fps, test_fps)
        sim_map = {c: s for c, s in zip(test_unique, unique_sims)}

        sims = test_df["SMILES_canonical"].fillna("_NA_").map(sim_map).to_numpy(dtype=float)
        fold_distributions[f"fold{int(fold)}"] = sims
        fold_summary = summarize_similarity(sims)
        fold_summary.update({
            "split": split_name,
            "fold": int(fold),
            "n_train_rows": int(len(train_df)),
            "n_test_rows": int(len(test_df)),
            "n_train_unique_canonical": int(len(train_unique)),
            "n_test_unique_canonical": int(len(test_unique)),
        })
        per_fold.append(fold_summary)
        for global_idx, (_, row) in zip(test_df.index.tolist(), test_df.iterrows()):
            can = row["SMILES_canonical"] if pd.notna(row["SMILES_canonical"]) else "_NA_"
            per_row.append(
                {
                    "split": split_name,
                    "fold": int(fold),
                    "row_index": int(global_idx),
                    "CID_std": row["CID_std"],
                    "SMILES_canonical": row["SMILES_canonical"],
                    "label_std": row["label_std"],
                    "nn_tanimoto_to_train": sim_map.get(can, np.nan),
                }
            )
    return pd.DataFrame(per_row), pd.DataFrame(per_fold), fold_distributions


def annotate_external_overlap(
    internal_compound_df: pd.DataFrame,
    external_compound_df: pd.DataFrame,
    external_graph_df: pd.DataFrame,
    strict_rule: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    internal_cid = set(internal_compound_df["CID_std"].dropna().astype(int).tolist())
    internal_raw = set(internal_compound_df["SMILES_raw"].dropna().astype(str).tolist())
    internal_can = set(internal_compound_df["SMILES_canonical"].dropna().astype(str).tolist())

    def annotate(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["overlap_cid"] = out["CID_std"].apply(lambda x: (pd.notna(x) and int(x) in internal_cid))
        out["overlap_raw_smiles"] = out["SMILES_raw"].astype(str).apply(lambda x: x in internal_raw)
        out["overlap_canonical_smiles"] = out["SMILES_canonical"].astype(str).apply(lambda x: x in internal_can)
        if strict_rule == "cid_or_canonical":
            out["strict_exclude"] = out["overlap_cid"] | out["overlap_canonical_smiles"]
        elif strict_rule == "cid_only":
            out["strict_exclude"] = out["overlap_cid"]
        elif strict_rule == "canonical_only":
            out["strict_exclude"] = out["overlap_canonical_smiles"]
        else:
            raise ValueError(strict_rule)
        out["strict_keep"] = ~out["strict_exclude"]
        return out

    ext_comp_annot = annotate(external_compound_df)
    ext_graph_annot = annotate(external_graph_df)

    strict_comp = ext_comp_annot.loc[ext_comp_annot["strict_keep"]].copy()
    strict_graph = ext_graph_annot.loc[ext_graph_annot["strict_keep"]].copy()

    overlap_summary = {
        "strict_overlap_rule": strict_rule,
        "external_compound_n": int(len(ext_comp_annot)),
        "external_compound_overlap_cid_n": int(ext_comp_annot["overlap_cid"].sum()),
        "external_compound_overlap_raw_smiles_n": int(ext_comp_annot["overlap_raw_smiles"].sum()),
        "external_compound_overlap_canonical_smiles_n": int(ext_comp_annot["overlap_canonical_smiles"].sum()),
        "external_compound_strict_keep_n": int(len(strict_comp)),
        "external_graph_n": int(len(ext_graph_annot)),
        "external_graph_overlap_cid_n": int(ext_graph_annot["overlap_cid"].sum()),
        "external_graph_overlap_raw_smiles_n": int(ext_graph_annot["overlap_raw_smiles"].sum()),
        "external_graph_overlap_canonical_smiles_n": int(ext_graph_annot["overlap_canonical_smiles"].sum()),
        "external_graph_strict_keep_n": int(len(strict_graph)),
    }
    return ext_comp_annot, strict_comp, ext_graph_annot, strict_graph, overlap_summary


def recompute_metrics(pred_df: pd.DataFrame, threshold: float) -> Dict[str, float]:
    y = pred_df["y_true"].astype(int).to_numpy()
    p = pred_df["p_ens"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        roc = np.nan
    else:
        roc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    f1 = float(f1_score(y, pred)) if len(np.unique(y)) > 1 else np.nan
    return {
        "n": int(len(pred_df)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
        "threshold": float(threshold),
        "roc_auc": roc,
        "pr_auc": ap,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }



def save_external_roc_pr_curves_strict(
    strict_pred_df: pd.DataFrame,
    out_dir: Path,
    prefix: str = "external_strict"
) -> None:
    if strict_pred_df is None or len(strict_pred_df) == 0:
        return

    y = strict_pred_df["y_true"].astype(int).to_numpy()
    p = strict_pred_df["p_ens"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return

    # ROC
    fpr, tpr, thr_roc = roc_curve(y, p)
    roc_auc = float(roc_auc_score(y, p))
    roc_df = pd.DataFrame({
        "analysis_set": "strict",
        "fpr": fpr.astype(float),
        "tpr": tpr.astype(float),
        "threshold": [float(x) if np.isfinite(x) else np.nan for x in thr_roc],
        "roc_auc": roc_auc,
        "n": int(len(strict_pred_df)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
    })
    roc_df.to_csv(out_dir / f"{prefix}_roc_curve_points.csv", index=False)

    plt.figure(figsize=(6.0, 5.0))
    plt.plot(fpr, tpr, linewidth=2, label=f"strict (ROC-AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("Strict external ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # PR
    precision, recall, thr_pr = precision_recall_curve(y, p)
    pr_auc = float(average_precision_score(y, p))
    pr_df = pd.DataFrame({
        "analysis_set": "strict",
        "recall": recall.astype(float),
        "precision": precision.astype(float),
        "threshold": [float(thr_pr[i]) if i < len(thr_pr) and np.isfinite(thr_pr[i]) else np.nan for i in range(len(precision))],
        "pr_auc": pr_auc,
        "n": int(len(strict_pred_df)),
        "n_pos": int(np.sum(y == 1)),
        "n_neg": int(np.sum(y == 0)),
    })
    pr_df.to_csv(out_dir / f"{prefix}_pr_curve_points.csv", index=False)

    plt.figure(figsize=(6.0, 5.0))
    plt.plot(recall, precision, linewidth=2, label=f"strict (PR-AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Strict external precision-recall curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_pr_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def best_f1_threshold(pred_df: pd.DataFrame) -> float:
    y = pred_df["y_true"].astype(int).to_numpy()
    p = pred_df["p_ens"].astype(float).to_numpy()
    prec, rec, thr = precision_recall_curve(y, p)
    # precision_recall_curve returns len(thr)=len(prec)-1
    if len(thr) == 0:
        return 0.5
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    idx = int(np.nanargmax(f1))
    return float(thr[idx])


def subset_external_predictions(pred_path: Path, strict_graph_df: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(pred_path)
    pred = add_standard_columns(
        pred,
        smiles_candidates=["SMILES", "smiles", "canonical_smiles"],
        cid_candidates=["CID", "cid", "PUBCHEM_CID"],
        label_candidates=["y_true", "Label", "label", "y"],
        dataset_name="external_predictions",
    )
    keep_keys = set(
        strict_graph_df.loc[:, ["CID_std", "SMILES_canonical"]]
        .drop_duplicates()
        .apply(lambda r: (None if pd.isna(r["CID_std"]) else int(r["CID_std"]), str(r["SMILES_canonical"])), axis=1)
        .tolist()
    )
    pred["strict_keep"] = pred.apply(
        lambda r: (None if pd.isna(r["CID_std"]) else int(r["CID_std"]), str(r["SMILES_canonical"])) in keep_keys,
        axis=1,
    )
    strict_pred = pred.loc[pred["strict_keep"]].copy()

    orig_best_thr = best_f1_threshold(pred)
    strict_best_thr = best_f1_threshold(strict_pred) if len(strict_pred) > 0 else 0.5
    metric_rows = []
    for tag, df_sub, thr in [
        ("external_original_thr0.5", pred, 0.5),
        ("external_original_thr_bestF1", pred, orig_best_thr),
        ("external_strict_thr0.5", strict_pred, 0.5),
        ("external_strict_thr_bestF1", strict_pred, strict_best_thr),
    ]:
        if len(df_sub) == 0:
            continue
        row = recompute_metrics(df_sub, threshold=thr)
        row["analysis_set"] = tag
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)

    # strict-only confusion outputs
    strict_conf_rows = []
    strict_pred_annot = strict_pred.copy()
    if len(strict_pred_annot) > 0:
        y = strict_pred_annot["y_true"].astype(int).to_numpy()
        p = strict_pred_annot["p_ens"].astype(float).to_numpy()
        for suffix, thr in [("thr0.5", 0.5), ("thr_bestF1", strict_best_thr)]:
            pred_label = (p >= thr).astype(int)
            conf_label = np.where((pred_label == 1) & (y == 1), "TP",
                           np.where((pred_label == 0) & (y == 0), "TN",
                           np.where((pred_label == 1) & (y == 0), "FP", "FN")))
            strict_pred_annot[f"pred_label_{suffix}"] = pred_label
            strict_pred_annot[f"confusion_{suffix}"] = conf_label
            strict_conf_rows.append({
                "analysis_set": f"external_strict_{suffix}",
                "threshold": float(thr),
                "n": int(len(strict_pred_annot)),
                "n_pos": int(np.sum(y == 1)),
                "n_neg": int(np.sum(y == 0)),
                "tp": int(np.sum(conf_label == "TP")),
                "tn": int(np.sum(conf_label == "TN")),
                "fp": int(np.sum(conf_label == "FP")),
                "fn": int(np.sum(conf_label == "FN")),
            })
    strict_conf_df = pd.DataFrame(strict_conf_rows)

    pred.to_csv(out_dir / "external_predictions_annotated.csv", index=False)
    strict_pred.to_csv(out_dir / "external_predictions_strict_subset.csv", index=False)
    strict_pred_annot.to_csv(out_dir / "external_predictions_strict_subset_annotated_confusion.csv", index=False)
    metrics_df.to_csv(out_dir / "external_prediction_metrics_original_vs_strict.csv", index=False)
    strict_conf_df.to_csv(out_dir / "external_confusion_strict_counts.csv", index=False)
    save_external_roc_pr_curves_strict(strict_pred, out_dir, prefix="external_strict")
    return pred, strict_pred, metrics_df


def maybe_subset_external_graph_pt(paths: Paths, strict_graph_df: pd.DataFrame) -> Optional[Path]:
    if paths.external_graph_pt is None or not paths.external_graph_pt.exists():
        return None
    try:
        import torch
        from torch.serialization import add_safe_globals
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
        from torch_geometric.data.storage import GlobalStorage

        add_safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage])
        try:
            obj = torch.load(paths.external_graph_pt, map_location="cpu")
        except Exception:
            obj = torch.load(paths.external_graph_pt, map_location="cpu", weights_only=False)
        if isinstance(obj, list):
            data_list = obj
            as_dict = False
        elif isinstance(obj, dict) and "data_list" in obj:
            data_list = obj["data_list"]
            as_dict = True
        else:
            data_list = list(obj)
            as_dict = False

        external_index = pd.read_csv(paths.external_graph_index_csv)
        external_index = add_standard_columns(
            external_index,
            smiles_candidates=["SMILES", "smiles", "SMILES_ISO", "PUBCHEM_EXT_DATASOURCE_SMILES_N"],
            cid_candidates=["CID", "cid", "PUBCHEM_CID"],
            label_candidates=["Label", "label", "y_true", "y"],
            dataset_name="external_graph_index",
        )
        if len(external_index) != len(data_list):
            print("[WARN] external graph index length and .pt data length differ; skipping .pt subsetting")
            return None

        keep_keys = set(
            strict_graph_df.loc[:, ["CID_std", "SMILES_canonical"]]
            .drop_duplicates()
            .apply(lambda r: (None if pd.isna(r["CID_std"]) else int(r["CID_std"]), str(r["SMILES_canonical"])), axis=1)
            .tolist()
        )
        keep_mask = external_index.apply(
            lambda r: (None if pd.isna(r["CID_std"]) else int(r["CID_std"]), str(r["SMILES_canonical"])) in keep_keys,
            axis=1,
        ).to_numpy()
        strict_data_list = [d for d, keep in zip(data_list, keep_mask) if keep]
        out_pt = paths.output_dir / "data_graph_external_strict.pt"
        if as_dict:
            obj["data_list"] = strict_data_list
            torch.save(obj, out_pt)
        else:
            torch.save(strict_data_list, out_pt)
        return out_pt
    except Exception as e:
        print(f"[WARN] Could not subset external .pt file: {e}")
        return None


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)

    # ---------- load and standardize ----------
    internal_compound = pd.read_csv(paths.internal_compound_csv)
    internal_graph = pd.read_csv(paths.internal_graph_index_csv)
    external_compound = pd.read_csv(paths.external_compound_csv)
    external_graph = pd.read_csv(paths.external_graph_index_csv)

    internal_compound = add_standard_columns(
        internal_compound,
        smiles_candidates=["SMILES", "smiles", "SMILES_ISO", "PUBCHEM_EXT_DATASOURCE_SMILES_N"],
        cid_candidates=["PUBCHEM_CID", "CID", "cid"],
        label_candidates=["label", "Label", "y_true", "y"],
        dataset_name="internal_compound_curated",
    )
    internal_graph = add_standard_columns(
        internal_graph,
        smiles_candidates=["SMILES", "smiles", "SMILES_ISO", "PUBCHEM_EXT_DATASOURCE_SMILES_N"],
        cid_candidates=["CID", "PUBCHEM_CID", "cid"],
        label_candidates=["Label", "label", "y_true", "y"],
        dataset_name="internal_graph_augmented",
    )
    external_compound = add_standard_columns(
        external_compound,
        smiles_candidates=["SMILES_ISO", "PUBCHEM_EXT_DATASOURCE_SMILES_N", "SMILES", "smiles"],
        cid_candidates=["PUBCHEM_CID", "CID", "cid"],
        label_candidates=["label", "Label", "y_true", "y"],
        dataset_name="external_compound_assay",
    )
    external_graph = add_standard_columns(
        external_graph,
        smiles_candidates=["SMILES", "smiles", "SMILES_ISO", "PUBCHEM_EXT_DATASOURCE_SMILES_N"],
        cid_candidates=["CID", "PUBCHEM_CID", "cid"],
        label_candidates=["Label", "label", "y_true", "y"],
        dataset_name="external_graph_graphable",
    )

    # ---------- dataset summaries ----------
    dataset_summary = pd.DataFrame(
        [
            summary_row("internal_compound_curated", internal_compound),
            summary_row("internal_graph_augmented", internal_graph),
            summary_row("external_compound_assay", external_compound),
            summary_row("external_graph_graphable", external_graph),
        ]
    )
    dataset_summary.to_csv(paths.output_dir / "dataset_assembly_summary.csv", index=False)

    # ---------- M2 scaffold split ----------
    m2_membership, m2_fold_summary = build_m2_scaffold_folds(internal_graph, n_folds=args.n_folds)
    m2_membership.to_csv(paths.output_dir / "split_membership_M2_scaffold5fold.csv", index=False)
    m2_fold_summary.to_csv(paths.output_dir / "split_summary_M2_scaffold5fold.csv", index=False)

    # ---------- M4 Butina-LCO split ----------
    m4_membership, m4_fold_summary, m4_cluster_summary, m4_assign_info = build_m4_lco_folds(
        internal_graph,
        radius=args.morgan_radius,
        nbits=args.morgan_nbits,
        butina_threshold=args.butina_threshold,
        n_folds=args.n_folds,
        min_pos_per_fold=args.min_pos_per_fold,
        min_neg_per_fold=args.min_neg_per_fold,
        fold_assign_trials=args.fold_assign_trials,
    )
    m4_membership.to_csv(paths.output_dir / "split_membership_M4_butina_lco5.csv", index=False)
    m4_fold_summary.to_csv(paths.output_dir / "split_summary_M4_butina_lco5.csv", index=False)
    m4_cluster_summary.to_csv(paths.output_dir / "cluster_summary_M4_butina_lco5.csv", index=False)
    with open(paths.output_dir / "cluster_assignment_info_M4_butina_lco5.json", "w", encoding="utf-8") as f:
        json.dump(m4_assign_info, f, ensure_ascii=False, indent=2)

    # ---------- split-wise similarity ----------
    m2_per_row, m2_per_fold, m2_dists = compute_fold_similarity(
        m2_membership,
        fold_col="m2_fold",
        split_name="M2_scaffold5fold",
        radius=args.morgan_radius,
        nbits=args.morgan_nbits,
    )
    m4_per_row, m4_per_fold, m4_dists = compute_fold_similarity(
        m4_membership,
        fold_col="m4_fold",
        split_name="M4_butina_lco5",
        radius=args.morgan_radius,
        nbits=args.morgan_nbits,
    )
    m2_per_row.to_csv(paths.output_dir / "similarity_per_row_M2_scaffold5fold.csv", index=False)
    m2_per_fold.to_csv(paths.output_dir / "similarity_summary_M2_scaffold5fold.csv", index=False)
    m4_per_row.to_csv(paths.output_dir / "similarity_per_row_M4_butina_lco5.csv", index=False)
    m4_per_fold.to_csv(paths.output_dir / "similarity_summary_M4_butina_lco5.csv", index=False)
    save_boxplot(m2_dists, paths.output_dir / "similarity_boxplot_M2_scaffold5fold.png", "M2 scaffold 5-fold: test-to-train similarity")
    save_boxplot(m4_dists, paths.output_dir / "similarity_boxplot_M4_butina_lco5.png", "M4 Butina-LCO: test-to-train similarity")

    # ---------- external overlap + strict external ----------
    ext_comp_annot, strict_comp, ext_graph_annot, strict_graph, overlap_summary = annotate_external_overlap(
        internal_compound_df=internal_compound,
        external_compound_df=external_compound,
        external_graph_df=external_graph,
        strict_rule=args.strict_overlap_rule,
    )
    ext_comp_annot.to_csv(paths.output_dir / "external_compound_overlap_annotated.csv", index=False)
    strict_comp.to_csv(paths.output_dir / "external_compound_strict.csv", index=False)
    ext_graph_annot.to_csv(paths.output_dir / "external_graph_overlap_annotated.csv", index=False)
    strict_graph.to_csv(paths.output_dir / "external_graph_strict.csv", index=False)
    with open(paths.output_dir / "external_overlap_summary.json", "w", encoding="utf-8") as f:
        json.dump(overlap_summary, f, ensure_ascii=False, indent=2)

    strict_graph_pt = maybe_subset_external_graph_pt(paths, strict_graph)

    # ---------- external similarity ----------
    internal_unique_can = pd.unique(internal_compound["SMILES_canonical"].fillna("_NA_"))
    fp_cache = {c: (None if c == "_NA_" else morgan_fp_from_smiles(c, args.morgan_radius, args.morgan_nbits)) for c in internal_unique_can}
    train_fps_unique = [fp_cache[c] for c in internal_unique_can]

    ext_orig_unique = pd.unique(ext_graph_annot["SMILES_canonical"].fillna("_NA_"))
    for c in ext_orig_unique:
        if c not in fp_cache:
            fp_cache[c] = None if c == "_NA_" else morgan_fp_from_smiles(c, args.morgan_radius, args.morgan_nbits)
    ext_orig_unique_sims = nearest_neighbor_similarity(train_fps_unique, [fp_cache[c] for c in ext_orig_unique])
    ext_orig_map = {c: s for c, s in zip(ext_orig_unique, ext_orig_unique_sims)}
    ext_orig_sims = ext_graph_annot["SMILES_canonical"].fillna("_NA_").map(ext_orig_map).to_numpy(dtype=float)

    ext_strict_unique = pd.unique(strict_graph["SMILES_canonical"].fillna("_NA_"))
    for c in ext_strict_unique:
        if c not in fp_cache:
            fp_cache[c] = None if c == "_NA_" else morgan_fp_from_smiles(c, args.morgan_radius, args.morgan_nbits)
    ext_strict_unique_sims = nearest_neighbor_similarity(train_fps_unique, [fp_cache[c] for c in ext_strict_unique])
    ext_strict_map = {c: s for c, s in zip(ext_strict_unique, ext_strict_unique_sims)}
    ext_strict_sims = strict_graph["SMILES_canonical"].fillna("_NA_").map(ext_strict_map).to_numpy(dtype=float)

    ext_graph_annot = ext_graph_annot.copy()
    ext_graph_annot["nn_tanimoto_to_internal"] = ext_orig_sims
    strict_graph = strict_graph.copy()
    strict_graph["nn_tanimoto_to_internal"] = ext_strict_sims
    ext_graph_annot.to_csv(paths.output_dir / "external_graph_overlap_annotated_with_similarity.csv", index=False)
    strict_graph.to_csv(paths.output_dir / "external_graph_strict_with_similarity.csv", index=False)

    external_similarity_summary = pd.DataFrame(
        [
            {"analysis_set": "external_original_vs_internal", **summarize_similarity(ext_orig_sims)},
            {"analysis_set": "external_strict_vs_internal", **summarize_similarity(ext_strict_sims)},
        ]
    )
    external_similarity_summary.to_csv(paths.output_dir / "similarity_summary_external_original_vs_strict.csv", index=False)
    save_boxplot(
        {
            "external_original": ext_orig_sims,
            "external_strict": ext_strict_sims,
        },
        paths.output_dir / "similarity_boxplot_external_original_vs_strict.png",
        "External-to-internal nearest-neighbor similarity",
    )
    save_histogram(ext_orig_sims, paths.output_dir / "similarity_hist_external_original.png", "External original vs internal")
    save_histogram(ext_strict_sims, paths.output_dir / "similarity_hist_external_strict.png", "External strict vs internal")

    # ---------- scaffold overlap ----------
    internal_scaffolds = set(internal_compound["SMILES_raw"].map(murcko_scaffold_from_smiles).dropna())
    ext_orig_scaffolds = set(ext_graph_annot["SMILES_raw"].map(murcko_scaffold_from_smiles).dropna())
    ext_strict_scaffolds = set(strict_graph["SMILES_raw"].map(murcko_scaffold_from_smiles).dropna())
    scaffold_summary = pd.DataFrame(
        [
            {
                "analysis_set": "external_original",
                "n_external_unique_scaffolds": len(ext_orig_scaffolds),
                "n_scaffolds_seen_in_internal": len(ext_orig_scaffolds & internal_scaffolds),
                "n_scaffolds_novel_vs_internal": len(ext_orig_scaffolds - internal_scaffolds),
                "pct_scaffolds_novel_vs_internal": (100.0 * len(ext_orig_scaffolds - internal_scaffolds) / len(ext_orig_scaffolds)) if ext_orig_scaffolds else np.nan,
            },
            {
                "analysis_set": "external_strict",
                "n_external_unique_scaffolds": len(ext_strict_scaffolds),
                "n_scaffolds_seen_in_internal": len(ext_strict_scaffolds & internal_scaffolds),
                "n_scaffolds_novel_vs_internal": len(ext_strict_scaffolds - internal_scaffolds),
                "pct_scaffolds_novel_vs_internal": (100.0 * len(ext_strict_scaffolds - internal_scaffolds) / len(ext_strict_scaffolds)) if ext_strict_scaffolds else np.nan,
            },
        ]
    )
    scaffold_summary.to_csv(paths.output_dir / "scaffold_overlap_summary_external_original_vs_strict.csv", index=False)

    # ---------- existing external predictions filtered to strict subset ----------
    external_metrics_df = pd.DataFrame()
    if paths.external_predictions_csv is not None and paths.external_predictions_csv.exists():
        _, _, external_metrics_df = subset_external_predictions(paths.external_predictions_csv, strict_graph, paths.output_dir)

    # ---------- response/manuscript helper tables ----------
    manuscript_table = pd.concat(
        [
            dataset_summary.assign(table_block="dataset_assembly"),
            m2_fold_summary.assign(table_block="M2_scaffold_folds"),
            m4_fold_summary.assign(table_block="M4_butina_folds"),
            external_similarity_summary.assign(table_block="external_similarity"),
            scaffold_summary.assign(table_block="external_scaffold_overlap"),
        ],
        axis=0,
        ignore_index=True,
        sort=False,
    )
    manuscript_table.to_csv(paths.output_dir / "comment5_manuscript_support_tables.csv", index=False)

    # ---------- markdown summary ----------
    md_lines = []
    md_lines.append("# Comment 5 analysis summary")
    md_lines.append("")
    md_lines.append("## Dataset assembly")
    md_lines.append(dataset_summary.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## External overlap summary")
    md_lines.append("```json")
    md_lines.append(json.dumps(overlap_summary, ensure_ascii=False, indent=2))
    md_lines.append("```")
    md_lines.append("")
    md_lines.append("## External similarity summary")
    md_lines.append(external_similarity_summary.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## M2 fold-wise similarity")
    md_lines.append(m2_per_fold.to_markdown(index=False))
    md_lines.append("")
    md_lines.append("## M4 fold-wise similarity")
    md_lines.append(m4_per_fold.to_markdown(index=False))
    md_lines.append("")
    if not external_metrics_df.empty:
        md_lines.append("## External prediction metrics (existing prediction file filtered to strict subset)")
        md_lines.append(external_metrics_df.to_markdown(index=False))
        md_lines.append("")
        md_lines.append("ROC figure (strict only): `external_strict_roc_curve.png`")
        md_lines.append("PR figure (strict only): `external_strict_pr_curve.png`")
        md_lines.append("Strict confusion counts: `external_confusion_strict_counts.csv`")
        md_lines.append("Strict row-level confusion annotations: `external_predictions_strict_subset_annotated_confusion.csv`")
        md_lines.append("")
    if strict_graph_pt is not None:
        md_lines.append(f"Strict external graph file: `{strict_graph_pt.name}`")
        md_lines.append("")

    with open(paths.output_dir / "comment5_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    # ---------- manifest ----------
    manifest = {
        "root": str(paths.root),
        "output_dir": str(paths.output_dir),
        "strict_external_graph_pt": str(strict_graph_pt) if strict_graph_pt is not None else None,
        "files": sorted([p.name for p in paths.output_dir.iterdir() if p.is_file()]),
        "notes": [
            "M2 split is reconstructed from graph-level augmented data using Murcko scaffold 5-fold round-robin assignment.",
            "M4 split is reconstructed from graph-level augmented data using Butina clustering plus constrained fold packing.",
            "Strict external set excludes compounds overlapping the internal curated set by the selected strict overlap rule.",
            "Strict external metrics are recalculated by filtering the existing external prediction CSV, not by re-running inference.",
        ],
    }
    with open(paths.output_dir / "comment5_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] Outputs written to: {paths.output_dir}")


if __name__ == "__main__":
    main()
