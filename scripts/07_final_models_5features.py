#!/usr/bin/env python3
"""
07 — Final modeling run with optimal 5-feature set (Experiment C).

Features: bpm, ibi, pnn20, rmssd, sd1
Models: RF, SVM-RBF, KNN, soft Voting Ensemble
Protocol: LOSO + SMOTE (train only), same hyperparameters as scripts/03.

Outputs:
  results/metrics/final_models_5features.json
  results/metrics/rf_feature_importance_5features.csv
  results/figures/16_final_confusion_matrices_5features.png
  results/figures/17_final_roc_curves_5features.png
  results/figures/18_final_rf_importance_5features.png

Run:
  python scripts/07_final_models_5features.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
FEATURES_CSV = ROOT / "data" / "processed" / "ephnogram_features.csv"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
OUT_JSON = METRICS_DIR / "final_models_5features.json"
OUT_IMP_CSV = METRICS_DIR / "rf_feature_importance_5features.csv"

FEATURE_COLS = ["bpm", "ibi", "pnn20", "rmssd", "sd1"]
LABEL_COL = "label"
GROUP_COL = "subject_id"
RANDOM_STATE = 42
SMOTE_K = 5
DPI = 300

MODEL_COLORS = {
    "Random Forest": "#2E7D32",
    "SVM RBF": "#1565C0",
    "KNN": "#6A1B9A",
    "Ensemble": "#C62828",
}
SHORT_NAMES = {
    "Random Forest": "RF",
    "SVM RBF": "SVM",
    "KNN": "KNN",
    "Ensemble": "Ens",
}


def build_models() -> dict:
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    svm = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=True,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                KNeighborsClassifier(n_neighbors=5, weights="distance", n_jobs=-1),
            ),
        ]
    )
    ensemble = VotingClassifier(
        estimators=[
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=200,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
            ),
            (
                "svm",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            SVC(
                                kernel="rbf",
                                C=1.0,
                                gamma="scale",
                                probability=True,
                                class_weight="balanced",
                                random_state=RANDOM_STATE,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "knn",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            KNeighborsClassifier(
                                n_neighbors=5, weights="distance", n_jobs=-1
                            ),
                        ),
                    ]
                ),
            ),
        ],
        voting="soft",
        n_jobs=-1,
    )
    return {
        "Random Forest": rf,
        "SVM RBF": svm,
        "KNN": knn,
        "Ensemble": ensemble,
    }


def _smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count <= 1:
        return X, y
    k = min(SMOTE_K, max(1, min_count - 1))
    return SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X, y)


def _proba_pos(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def loso_evaluate(df: pd.DataFrame, model, model_name: str) -> dict:
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy(dtype=int)
    groups = df[GROUP_COL].astype(str).to_numpy()

    fold_rows: list[dict] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_proba_all: list[float] = []
    per_subject: list[dict] = []

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        held_out = str(groups[test_idx][0])
        X_tr, y_tr = _smote(X[train_idx], y[train_idx])
        X_te, y_te = X[test_idx], y[test_idx]

        clf = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_proba = _proba_pos(clf, X_te)

        acc = float(accuracy_score(y_te, y_pred))
        f1m = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        f1w = float(f1_score(y_te, y_pred, average="weighted", zero_division=0))

        fold_rows.append(
            {
                "subject": held_out,
                "n_test": int(len(y_te)),
                "accuracy": acc,
                "f1_macro": f1m,
                "f1_weighted": f1w,
            }
        )
        per_subject.append({"subject": held_out, "accuracy": acc, "n_test": int(len(y_te))})
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(np.asarray(y_pred).tolist())
        y_proba_all.extend(np.asarray(y_proba, dtype=float).tolist())

    summary: dict = {
        "model": model_name,
        "n_features": len(FEATURE_COLS),
        "n_folds": len(fold_rows),
    }
    for m in ("accuracy", "f1_macro", "f1_weighted"):
        vals = np.asarray([r[m] for r in fold_rows], dtype=float)
        summary[f"{m}_mean"] = float(np.mean(vals))
        summary[f"{m}_std"] = float(np.std(vals, ddof=0))

    y_true_arr = np.asarray(y_true_all, dtype=int)
    y_proba_arr = np.asarray(y_proba_all, dtype=float)
    y_pred_arr = np.asarray(y_pred_all, dtype=int)
    pooled_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
    fpr, tpr, _ = roc_curve(y_true_arr, y_proba_arr)

    summary["roc_auc_pooled"] = pooled_auc
    summary["confusion_matrix"] = confusion_matrix(
        y_true_arr, y_pred_arr, labels=[0, 1]
    ).tolist()
    summary["folds"] = fold_rows
    summary["per_subject_accuracy"] = per_subject
    summary["_roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": pooled_auc}
    return summary


def print_table(results: list[dict]) -> None:
    print("\nModel | Features | Accuracy     | F1-macro     | F1-weighted  | ROC-AUC")
    print("-" * 78)
    for r in results:
        short = SHORT_NAMES.get(r["model"], r["model"])
        print(
            f"{short:<5} | "
            f"{r['n_features']:<8} | "
            f"{r['accuracy_mean']:.3f}±{r['accuracy_std']:.3f} | "
            f"{r['f1_macro_mean']:.3f}±{r['f1_macro_std']:.3f} | "
            f"{r['f1_weighted_mean']:.3f}±{r['f1_weighted_std']:.3f} | "
            f"{r['roc_auc_pooled']:.3f}"
        )


def plot_confusion_grid(results: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    for ax, r in zip(axes.ravel(), results):
        cm = np.asarray(r["confusion_matrix"], dtype=int)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["REST", "STRESS"],
            yticklabels=["REST", "STRESS"],
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(
            f"{r['model']}\n"
            f"Acc={r['accuracy_mean']:.3f}  F1m={r['f1_macro_mean']:.3f}"
        )
    fig.suptitle(
        "Final models — LOSO confusion matrices (5 features: bpm, ibi, pnn20, rmssd, sd1)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(results: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    for r in results:
        roc = r["_roc"]
        name = r["model"]
        ax.plot(
            roc["fpr"],
            roc["tpr"],
            color=MODEL_COLORS.get(name, "gray"),
            linewidth=2,
            label=f"{name} (AUC={roc['auc']:.3f})",
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Final models — pooled ROC (optimal 5 features)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_rf_importance(df: pd.DataFrame, out: Path) -> pd.DataFrame:
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy(dtype=int)
    X_bal, y_bal = _smote(X, y)
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    rf.fit(X_bal, y_bal)
    imp = pd.DataFrame(
        {"feature": FEATURE_COLS, "importance": rf.feature_importances_}
    ).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(
        imp["feature"],
        imp["importance"],
        color="#2E7D32",
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_xlabel("Mean decrease in impurity (RF importance)")
    ax.set_ylabel("Feature")
    ax.set_title("Final RF feature importance — optimal 5 features (SMOTE-balanced)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    ranked = imp.sort_values("importance", ascending=False)
    ranked.to_csv(OUT_IMP_CSV, index=False)
    return ranked


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    if not FEATURES_CSV.exists():
        print(f"ERROR: missing {FEATURES_CSV}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(FEATURES_CSV)
    n_rest = int((df[LABEL_COL] == 0).sum())
    n_stress = int((df[LABEL_COL] == 1).sum())
    n_subj = int(df[GROUP_COL].nunique())
    print(
        f"Loaded {len(df)} windows | rest={n_rest} stress={n_stress} | "
        f"subjects={n_subj}"
    )
    print(f"Optimal features ({len(FEATURE_COLS)}): {FEATURE_COLS}\n")

    models = build_models()
    results: list[dict] = []
    for name, model in models.items():
        print(f"=== LOSO: {name} ===")
        res = loso_evaluate(df, model, name)
        results.append(res)
        print(
            f"  Acc={res['accuracy_mean']:.3f}±{res['accuracy_std']:.3f}  "
            f"F1m={res['f1_macro_mean']:.3f}±{res['f1_macro_std']:.3f}  "
            f"AUC={res['roc_auc_pooled']:.3f}"
        )

    print_table(results)

    best = max(results, key=lambda r: r["f1_macro_mean"])
    print(f"\nBest by F1-macro: {best['model']} = {best['f1_macro_mean']:.3f}")

    # Figures
    p_cm = FIGURES_DIR / "16_final_confusion_matrices_5features.png"
    plot_confusion_grid(results, p_cm)
    print(f"Saved -> {p_cm}")

    p_roc = FIGURES_DIR / "17_final_roc_curves_5features.png"
    plot_roc_curves(results, p_roc)
    print(f"Saved -> {p_roc}")

    p_imp = FIGURES_DIR / "18_final_rf_importance_5features.png"
    imp = plot_rf_importance(df, p_imp)
    print(f"Saved -> {p_imp}")
    print(f"Saved -> {OUT_IMP_CSV}")
    print("RF importance (5 features):\n", imp.to_string(index=False))

    # JSON without internal ROC arrays duplicated as private key — keep roc for viz reuse
    models_out = []
    for r in results:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        row["roc_curve"] = r["_roc"]
        models_out.append(row)

    payload = {
        "dataset": "ephnogram",
        "feature_set": "optimal_5_experiment_C",
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "n_samples": int(len(df)),
        "n_rest": n_rest,
        "n_stress": n_stress,
        "n_subjects": n_subj,
        "majority_baseline_accuracy": float(n_stress / len(df)),
        "best_model_f1_macro": best["model"],
        "protocol": "LOSO + SMOTE(train only); RF/SVM class_weight=balanced; soft Voting",
        "models": models_out,
        "figures": [
            str(p_cm),
            str(p_roc),
            str(p_imp),
        ],
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics -> {OUT_JSON}")


if __name__ == "__main__":
    main()
