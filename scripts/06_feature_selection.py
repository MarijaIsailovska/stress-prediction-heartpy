#!/usr/bin/env python3
"""
06 — Feature engineering + LOSO RF feature-set comparison (EPHNOGRAM).

Creates data/processed/ephnogram_features_engineered.csv and compares:
  A) all 11 original features
  B) reduced 7
  C) minimal 5
  D) engineered 9

Run:
  python scripts/06_feature_selection.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

# ---------------------------------------------------------------------------
IN_CSV = ROOT / "data" / "processed" / "ephnogram_features.csv"
OUT_CSV = ROOT / "data" / "processed" / "ephnogram_features_engineered.csv"
OUT_JSON = ROOT / "results" / "metrics" / "feature_selection_comparison.json"

LABEL_COL = "label"
GROUP_COL = "subject_id"
RANDOM_STATE = 42
SMOTE_K = 5
F1_DROP_THRESHOLD = 0.02  # "no significant drop" vs best / baseline A

FEATURES_A = [
    "bpm",
    "ibi",
    "sdnn",
    "rmssd",
    "sdsd",
    "pnn20",
    "pnn50",
    "hr_mad",
    "sd1",
    "sd2",
    "breathing_rate",
]
FEATURES_B = ["bpm", "ibi", "pnn20", "rmssd", "sd1", "hr_mad", "sdnn"]
FEATURES_C = ["bpm", "ibi", "pnn20", "rmssd", "sd1"]
FEATURES_D = [
    "bpm",
    "ibi",
    "sdnn",
    "rmssd",
    "pnn20",
    "hr_mad",
    "sd1_sd2_ratio",
    "baevsky_proxy",
    "pnn20_normalized",
]

EXPERIMENTS = [
    ("A (all 11)", FEATURES_A),
    ("B (reduced 7)", FEATURES_B),
    ("C (minimal 5)", FEATURES_C),
    ("D (engineered 9)", FEATURES_D),
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sd1_sd2_ratio, baevsky_proxy, pnn20_normalized."""
    out = df.copy()
    sd2 = out["sd2"].replace(0, np.nan)
    out["sd1_sd2_ratio"] = out["sd1"] / sd2
    out["baevsky_proxy"] = out["hr_mad"] / (out["rmssd"] + 1e-6)
    out["pnn20_normalized"] = out["pnn20"] / (out["ibi"] / 1000.0)

    # Finite check — replace inf/nan from division edge cases
    for col in ("sd1_sd2_ratio", "baevsky_proxy", "pnn20_normalized"):
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        if out[col].isna().any():
            # Rare: fill with column median of finite values within class-agnostic median
            med = float(out[col].median(skipna=True))
            out[col] = out[col].fillna(med)

    return out


def _smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count <= 1:
        return X, y
    k = min(SMOTE_K, max(1, min_count - 1))
    return SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X, y)


def loso_rf(df: pd.DataFrame, feature_cols: list[str], experiment_name: str) -> dict:
    """LOSO RF with SMOTE on train only; pooled ROC-AUC."""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{experiment_name}: missing columns {missing}")

    X = df[feature_cols].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy(dtype=int)
    groups = df[GROUP_COL].astype(str).to_numpy()

    fold_acc: list[float] = []
    fold_f1: list[float] = []
    y_true_all: list[int] = []
    y_proba_all: list[float] = []

    rf_template = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )

    print(f"\n=== {experiment_name} | n_features={len(feature_cols)} ===")
    print(f"  features: {feature_cols}")

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        X_tr, y_tr = _smote(X[train_idx], y[train_idx])
        X_te, y_te = X[test_idx], y[test_idx]
        clf = clone(rf_template)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            proba = clf.predict_proba(X_te)
            classes = list(clf.classes_)
            y_proba = proba[:, classes.index(1)] if 1 in classes else proba[:, -1]

        fold_acc.append(float(accuracy_score(y_te, y_pred)))
        fold_f1.append(float(f1_score(y_te, y_pred, average="macro", zero_division=0)))
        y_true_all.extend(y_te.tolist())
        y_proba_all.extend(np.asarray(y_proba, dtype=float).tolist())

    acc_mean = float(np.mean(fold_acc))
    acc_std = float(np.std(fold_acc, ddof=0))
    f1_mean = float(np.mean(fold_f1))
    f1_std = float(np.std(fold_f1, ddof=0))
    y_true_arr = np.asarray(y_true_all, dtype=int)
    y_proba_arr = np.asarray(y_proba_all, dtype=float)
    try:
        pooled_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
    except ValueError:
        pooled_auc = float("nan")

    print(
        f"  Acc={acc_mean:.3f}±{acc_std:.3f}  "
        f"F1m={f1_mean:.3f}±{f1_std:.3f}  "
        f"AUC_pooled={pooled_auc:.3f}"
    )

    return {
        "experiment": experiment_name,
        "n_features": len(feature_cols),
        "features": feature_cols,
        "accuracy_mean": acc_mean,
        "accuracy_std": acc_std,
        "f1_macro_mean": f1_mean,
        "f1_macro_std": f1_std,
        "roc_auc_pooled": pooled_auc,
        "fold_accuracy": fold_acc,
        "fold_f1_macro": fold_f1,
    }


def print_table(results: list[dict]) -> None:
    print("\nExperiment        | N features | Accuracy      | F1-macro      | ROC-AUC")
    print("-" * 78)
    for r in results:
        print(
            f"{r['experiment']:<17} | "
            f"{r['n_features']:<10} | "
            f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f}  | "
            f"{r['f1_macro_mean']:.3f}±{r['f1_macro_std']:.3f}  | "
            f"{r['roc_auc_pooled']:.3f}"
        )


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    if not IN_CSV.exists():
        print(f"ERROR: missing {IN_CSV}", file=sys.stderr)
        sys.exit(1)

    df0 = pd.read_csv(IN_CSV)
    df = engineer_features(df0)
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved engineered features -> {OUT_CSV}")
    print(
        "Engineered cols: "
        f"sd1_sd2_ratio, baevsky_proxy, pnn20_normalized "
        f"(n={len(df)})"
    )

    results = [loso_rf(df, feats, name) for name, feats in EXPERIMENTS]
    print_table(results)

    # Winner + practical conclusion
    best = max(results, key=lambda r: r["f1_macro_mean"])
    baseline = results[0]  # A
    print(f"\nWinner (F1-macro): {best['experiment']} = {best['f1_macro_mean']:.3f}")

    # Minimum features without F1 drop > 0.02 vs baseline A
    eligible = [
        r
        for r in results
        if (baseline["f1_macro_mean"] - r["f1_macro_mean"]) <= F1_DROP_THRESHOLD
    ]
    # Prefer fewest features among those within threshold of A;
    # if engineered beats A, still prefer fewest within threshold of best?
    # Spec: "minimum features needed without significant performance drop
    # (threshold: F1-macro drop < 0.02)" — interpret vs baseline A.
    if not eligible:
        eligible = [baseline]
    recommended = min(eligible, key=lambda r: r["n_features"])
    # If multiple same n, pick higher F1
    same_n = [r for r in eligible if r["n_features"] == recommended["n_features"]]
    recommended = max(same_n, key=lambda r: r["f1_macro_mean"])

    print(
        f"Practical recommendation (F1 drop vs A ≤ {F1_DROP_THRESHOLD}): "
        f"{recommended['experiment']} "
        f"({recommended['n_features']} features, "
        f"F1m={recommended['f1_macro_mean']:.3f})"
    )

    payload = {
        "input_csv": str(IN_CSV),
        "engineered_csv": str(OUT_CSV),
        "f1_drop_threshold": F1_DROP_THRESHOLD,
        "winner_f1_macro": best["experiment"],
        "recommended_min_features": recommended["experiment"],
        "experiments": results,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
