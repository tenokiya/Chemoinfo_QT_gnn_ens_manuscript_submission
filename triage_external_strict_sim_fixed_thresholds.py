# ============================================================
# Strict external triage simulation (GREEN / YELLOW / RED) — Chemoinfo_QT
#
# Purpose:
#   - Fix the triage rules using only the thresholds (JSON) defined in the internal data
#   - Using the prediction CSV from the strict external dataset as input,
#     aggregate and output the “simulated results” of that triage
#
# Expected Input (Example candidates for strict external prediction CSV):
#   - strict_external_best_predictions.csv
#   - external_predictions_strict_subset.csv
#   - predictions_external_strict*.csv
#
# 3-way classification rules:
#   - GREEN : p <  thr_green
#   - YELLOW: thr_green <= p < thr_red
#   - RED   : p >= thr_red
#
# Output (under OUTDIR):
#   - triage_assignments_strict_external_<GroupTag>.csv
#   - triage_summary_strict_external_<GroupTag>.csv
#   - triage_rule_and_kpi_strict_external_<GroupTag>.json
#   - triage_confusion_reference_strict_external_<GroupTag>.csv
# ============================================================

Translated with DeepL.com (free version)

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# Defaults
# =========================
DEFAULT_ROOT = "/content/drive/MyDrive/Chemoinfo_QT"
DEFAULT_INTERNAL_THR_JSON = f"{DEFAULT_ROOT}/reports_internal_threshold_search/threshold_targets_internal_CID.json"
DEFAULT_OUTDIR = f"{DEFAULT_ROOT}/reports_triage_external_strict_fixed"


# =========================
# Utilities
# =========================
def discover_strict_external_csv(root: str) -> str:
    """
    strict external 用の予測CSVを自動探索する。
    優先順：
      1) strict_external_best_predictions.csv
      2) external_predictions_strict_subset.csv
      3) predictions_external_strict*.csv
      4) strict を含む prediction csv
    """
    root_p = Path(root)

    preferred_names = [
        "strict_external_best_predictions.csv",
        "external_predictions_strict_subset.csv",
        "strict_external_predictions.csv",
    ]

    hits = []
    for name in preferred_names:
        hits.extend([str(p) for p in root_p.rglob(name)])
    if hits:
        hits = sorted(set(hits), key=lambda p: os.path.getmtime(p), reverse=True)
        return hits[0]

    patterns = [
        "predictions_external_strict*.csv",
        "*strict*external*pred*.csv",
        "*external*strict*pred*.csv",
        "*strict*prediction*.csv",
        "*strict*.csv",
    ]

    candidates = []
    for pat in patterns:
        candidates.extend([str(p) for p in root_p.rglob(pat)])

    # prediction 系CSVを優先
    filtered = []
    for p in candidates:
        name = Path(p).name.lower()
        if "pred" in name or "prediction" in name or "strict_external" in name:
            filtered.append(p)

    if filtered:
        filtered = sorted(set(filtered), key=lambda p: os.path.getmtime(p), reverse=True)
        return filtered[0]

    raise FileNotFoundError(
        "Strict external predictions CSV was not found. Please pass --strict_external_csv explicitly."
    )


def _pick_first_existing(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def detect_columns(df: pd.DataFrame, prob_col_hint: str = "p_ens"):
    """
    予測CSVから必要列を推定する。
    戻り値：
      (y_col, p_col, cid_col, smiles_col)
    """
    cols = list(df.columns)

    # y
    y_col = _pick_first_existing(
        cols,
        ["y_true", "Label", "label", "Y", "y", "target", "Target", "class", "Class"]
    )
    if y_col is None:
        raise ValueError(f"Could not detect label column. Available columns: {cols}")

    # prob
    if prob_col_hint in cols:
        p_col = prob_col_hint
    else:
        p_col = _pick_first_existing(
            cols,
            [
                "p_ens", "p_final", "p_weighted", "p_avg", "p_mean",
                "pred_prob", "pred_probability", "pred_score",
                "prob", "probability", "score", "p", "pred", "prediction"
            ]
        )
    if p_col is None:
        p_col = next((c for c in cols if str(c).lower().startswith("p_")), None)
    if p_col is None:
        p_col = next((c for c in cols if any(k in str(c).lower() for k in ["pred_prob", "prob", "probability"])), None)
    if p_col is None:
        raise ValueError(f"Could not detect probability column. Available columns: {cols}")

    # group key (CID preferred)
    cid_col = _pick_first_existing(
        cols,
        ["CID", "cid", "PubChem_CID", "pubchem_cid", "compound_cid", "drug_cid", "group_key"]
    )
    smiles_col = _pick_first_existing(
        cols,
        ["SMILES", "smiles", "smiles_orig", "smiles_original", "canonical_smiles"]
    )

    return y_col, p_col, cid_col, smiles_col


def load_thresholds_from_json(json_path: str):
    """
    内部閾値JSONから triage 用閾値を抽出する。
    期待：
      - precision系 threshold → thr_red（高い閾値）
      - recall系 threshold     → thr_green（低い閾値）
    """
    jp = Path(json_path)
    if not jp.exists():
        raise FileNotFoundError(f"internal_thr_json not found: {json_path}")

    with open(jp, "r", encoding="utf-8") as f:
        obj = json.load(f)

    meta = {"json_path": str(jp)}

    if "thr_green" in obj and "thr_red" in obj:
        thr_green = float(obj["thr_green"])
        thr_red = float(obj["thr_red"])
        meta["source"] = "direct_keys"
        return thr_green, thr_red, meta

    selected = obj.get("selected", {})
    if not isinstance(selected, dict) or not selected:
        raise ValueError("threshold json missing 'selected' dict and also missing direct thr_green/thr_red keys.")

    recall_dict = None
    for k, v in selected.items():
        if isinstance(v, dict) and ("Recall" in k or "recall" in k):
            recall_dict = v
            meta["recall_key"] = k
            break

    prec_dict = None
    for k, v in selected.items():
        if isinstance(v, dict) and ("Precision" in k or "precision" in k):
            prec_dict = v
            meta["precision_key"] = k
            break

    if recall_dict is None or prec_dict is None:
        raise ValueError(f"Could not locate recall/precision entries in json['selected']. Keys={list(selected.keys())}")

    if "Threshold" not in recall_dict or "Threshold" not in prec_dict:
        raise ValueError("selected recall/precision dict must contain 'Threshold'.")

    thr_green = float(recall_dict["Threshold"])
    thr_red = float(prec_dict["Threshold"])

    meta["source"] = "selected_dict"
    meta["thr_green_from"] = recall_dict.get("criterion", "recall_threshold")
    meta["thr_red_from"] = prec_dict.get("criterion", "precision_threshold_or_fallback")

    return thr_green, thr_red, meta


def aggregate_drug_level(df: pd.DataFrame,
                         y_col: str,
                         p_col: str,
                         key_col: str,
                         prob_agg: str = "mean",
                         y_agg: str = "max") -> pd.DataFrame:
    """
    group_key（CID等）で drug-level に集約する。
      - y_true: max（保守的） or mode
      - p: mean or median
    """
    d = df.copy()

    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[p_col] = pd.to_numeric(d[p_col], errors="coerce")
    d = d.dropna(subset=[y_col, p_col, key_col])

    d[y_col] = d[y_col].astype(int)
    d = d[d[y_col].isin([0, 1])].copy()

    if y_agg == "mode":
        def agg_mode(arr):
            vals, cnts = np.unique(arr, return_counts=True)
            pick = vals[np.argmax(cnts)]
            if len(vals) > 1:
                print(f"[WARN] y_true conflict in a group -> using mode={pick} (counts {dict(zip(vals, cnts))})")
            return int(pick)
        y_fn = agg_mode
    else:
        y_fn = "max"

    if prob_agg == "median":
        p_fn = lambda x: float(np.median(pd.to_numeric(x, errors="coerce").dropna().values))
    else:
        p_fn = lambda x: float(np.mean(pd.to_numeric(x, errors="coerce").dropna().values))

    g = (
        d.groupby(key_col, dropna=False, as_index=False)
         .agg(
            y_true=(y_col, y_fn),
            p_ens=(p_col, p_fn),
            n_instances=(p_col, "count")
         )
    )
    return g


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def metrics_from_counts(tp, fp, tn, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision_pos": precision,
        "Recall_pos": recall,
        "Specificity": specificity,
        "NPV": npv,
        "Accuracy": acc,
        "F1_pos": f1
    }


def summarize_tiers(g: pd.DataFrame, thr_green: float, thr_red: float):
    d = g.copy()
    p = d["p_ens"].to_numpy(dtype=float)

    tier = np.where(p < thr_green, "GREEN",
            np.where(p >= thr_red, "RED", "YELLOW"))
    d["tier"] = tier

    tier_df = (
        d.groupby("tier", as_index=False)
         .agg(
            N=("tier", "size"),
            Pos=("y_true", "sum"),
            Neg=("y_true", lambda x: int((np.array(x) == 0).sum())),
            MeanProb=("p_ens", "mean"),
            MedianProb=("p_ens", "median"),
            MeanInstances=("n_instances", "mean")
         )
    )
    tier_df["PosRate"] = tier_df["Pos"] / tier_df["N"]

    y_true = d["y_true"].to_numpy(dtype=int)

    # RED only positive
    y_pred_red = (d["tier"].to_numpy() == "RED").astype(int)
    tp, fp, tn, fn = confusion_counts(y_true, y_pred_red)
    kpi_red = metrics_from_counts(tp, fp, tn, fn)
    kpi_red["Definition"] = "Predict positive = RED only"

    # YELLOW + RED positive
    y_pred_not_green = (d["tier"].to_numpy() != "GREEN").astype(int)
    tp2, fp2, tn2, fn2 = confusion_counts(y_true, y_pred_not_green)
    kpi_not_green = metrics_from_counts(tp2, fp2, tn2, fn2)
    kpi_not_green["Definition"] = "Predict positive = (YELLOW or RED)"

    green = d[d["tier"] == "GREEN"].copy()
    if len(green) > 0:
        green_pos = int(green["y_true"].sum())
        green_neg = int((green["y_true"] == 0).sum())
        green_pos_rate = green_pos / len(green)
    else:
        green_pos = green_neg = 0
        green_pos_rate = np.nan

    pos_total = int(y_true.sum())
    pos_red = int(d.loc[d["tier"] == "RED", "y_true"].sum())
    pos_yellow_red = int(d.loc[d["tier"].isin(["YELLOW", "RED"]), "y_true"].sum())
    capture = {
        "Pos_total": pos_total,
        "Pos_in_RED": pos_red,
        "Pos_in_YELLOW_or_RED": pos_yellow_red,
        "Capture_RED": (pos_red / pos_total) if pos_total else np.nan,
        "Capture_YELLOW_or_RED": (pos_yellow_red / pos_total) if pos_total else np.nan,
        "GREEN_Pos": green_pos,
        "GREEN_Neg": green_neg,
        "GREEN_PosRate": green_pos_rate
    }

    # reference confusion table
    conf_rows = [
        {"Scenario": "RED_only", **kpi_red},
        {"Scenario": "YELLOW_or_RED", **kpi_not_green},
    ]
    conf_df = pd.DataFrame(conf_rows)

    kpi = {"RED_only": kpi_red, "YELLOW_or_RED": kpi_not_green, "capture": capture}
    return tier_df.sort_values("tier"), conf_df, kpi, d


# =========================
# Main
# =========================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--strict_external_csv", default=None, help="strict external predictions csv (if omitted, auto-discover)")
    ap.add_argument("--internal_thr_json", default=DEFAULT_INTERNAL_THR_JSON, help="threshold_targets_internal_CID.json")
    ap.add_argument("--thr_green", type=float, default=None, help="override GREEN upper threshold")
    ap.add_argument("--thr_red", type=float, default=None, help="override RED lower threshold")
    ap.add_argument("--prob_col", default="p_ens", help="probability column (default: p_ens)")
    ap.add_argument("--prob_agg", choices=["mean", "median"], default="mean")
    ap.add_argument("--y_agg", choices=["max", "mode"], default="max")
    ap.add_argument("--outdir", default=None, help=f"output directory (default: {DEFAULT_OUTDIR})")

    args, unknown = ap.parse_known_args(args=argv)
    if unknown:
        print("[INFO] Ignoring unknown argv from notebook kernel:", unknown)

    root = args.root
    outdir = args.outdir or DEFAULT_OUTDIR
    os.makedirs(outdir, exist_ok=True)

    strict_external_csv = args.strict_external_csv or discover_strict_external_csv(root)
    if not Path(strict_external_csv).exists():
        raise FileNotFoundError(f"not found: {strict_external_csv}")

    thr_green, thr_red, thr_meta = load_thresholds_from_json(args.internal_thr_json)
    if args.thr_green is not None:
        thr_green = float(args.thr_green)
        thr_meta["thr_green_override"] = True
    if args.thr_red is not None:
        thr_red = float(args.thr_red)
        thr_meta["thr_red_override"] = True

    if not (thr_green < thr_red):
        raise ValueError(f"threshold order invalid: thr_green={thr_green}, thr_red={thr_red}")

    print("[INFO] Strict external CSV:", strict_external_csv)
    print("[INFO] Thresholds fixed:", {"thr_green": thr_green, "thr_red": thr_red})
    print("[INFO] Threshold meta:", thr_meta)

    df = pd.read_csv(strict_external_csv)
    y_col, p_col, cid_col, smiles_col = detect_columns(df, prob_col_hint=args.prob_col)

    if cid_col is not None:
        key_col = cid_col
        group_tag = "CID"
    elif smiles_col is not None:
        key_col = smiles_col
        group_tag = "SMILES"
    else:
        raise ValueError("Neither CID nor SMILES column found for grouping.")

    print("[INFO] Detected columns:", {"y_col": y_col, "p_col": p_col, "group_key": f"{group_tag}({key_col})"})

    g = aggregate_drug_level(
        df=df,
        y_col=y_col,
        p_col=p_col,
        key_col=key_col,
        prob_agg=args.prob_agg,
        y_agg=args.y_agg,
    )

    print(f"[INFO] Drug-level N={len(g)} | Pos={int(g['y_true'].sum())} | Neg={int((g['y_true']==0).sum())}")

    tier_df, conf_df, kpi, assign = summarize_tiers(g, thr_green=thr_green, thr_red=thr_red)

    assign_csv = f"{outdir}/triage_assignments_strict_external_{group_tag}.csv"
    assign.to_csv(assign_csv, index=False)

    summary_csv = f"{outdir}/triage_summary_strict_external_{group_tag}.csv"
    tier_df.to_csv(summary_csv, index=False)

    conf_csv = f"{outdir}/triage_confusion_reference_strict_external_{group_tag}.csv"
    conf_df.to_csv(conf_csv, index=False)

    rule_json = f"{outdir}/triage_rule_and_kpi_strict_external_{group_tag}.json"
    with open(rule_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "strict_external_csv": strict_external_csv,
                "group_tag": group_tag,
                "prob_col": p_col,
                "prob_agg": args.prob_agg,
                "y_agg": args.y_agg,
                "thresholds": {"thr_green": thr_green, "thr_red": thr_red},
                "threshold_meta": thr_meta,
                "tier_summary": tier_df.to_dict(orient="records"),
                "kpi": kpi,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Tier summary (strict external) ===")
    print(tier_df.to_string(index=False))
    print("\n=== KPI (reference) ===")
    print(json.dumps(kpi, ensure_ascii=False, indent=2))
    print("\n[SAVE] assignments:", assign_csv)
    print("[SAVE] summary    :", summary_csv)
    print("[SAVE] confusion  :", conf_csv)
    print("[SAVE] kpi json   :", rule_json)
    print("\nDone.")


if __name__ == "__main__":
    main()
