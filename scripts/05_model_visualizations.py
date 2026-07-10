#!/usr/bin/env python3
"""
05 — Model evaluation visualizations for EPHNOGRAM binary LOSO results.

Reads results/metrics/ephnogram_binary_loso.json (confusion matrices, F1,
per-subject accuracy) and recomputes pooled ROC curves + RF feature importance.

Outputs (DPI=300) under results/figures/:
  11_confusion_matrices_grid.png
  12_roc_curves_pooled.png
  13_per_subject_accuracy.png
  14_f1_macro_comparison.png
  15_rf_feature_importance.png

Run:
  python scripts/05_model_visualizations.py
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
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
FEATURES_CSV = ROOT / "data" / "processed" / "ephnogram_features.csv"
METRICS_JSON = ROOT / "results" / "metrics" / "ephnogram_binary_loso.json"
FIGURES_DIR = ROOT / "results" / "figures"
DPI = 300

FEATURE_COLS = [
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
LABEL_COL = "label"
GROUP_COL = "subject_id"
RANDOM_STATE = 42
SMOTE_K = 5
MAJORITY_BASELINE = 0.77

MODEL_COLORS = {
    "Random Forest": "#2E7D32",
    "SVM RBF": "#1565C0",
    "KNN": "#6A1B9A",
    "Ensemble": "#C62828",
}


def load_metrics() -> dict:
    if not METRICS_JSON.exists():
        raise FileNotFoundError(
            f"Missing {METRICS_JSON}. Run scripts/03_train_evaluate.py first."
        )
    text = METRICS_JSON.read_text(encoding="utf-8").replace("NaN", "null")
    return json.loads(text)


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


def _smote(X, y):
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count <= 1:
        return X, y
    k = min(SMOTE_K, max(1, min_count - 1))
    return SMOTE(k_neighbors=k, random_state=RANDOM_STATE).fit_resample(X, y)


def _proba_pos(model, X):
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def collect_pooled_scores(df: pd.DataFrame) -> dict[str, dict]:
    """Re-run LOSO to collect y_true / y_proba for pooled ROC curves."""
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy(dtype=int)
    groups = df[GROUP_COL].astype(str).to_numpy()
    models = build_models()
    out: dict[str, dict] = {}

    for name, model in models.items():
        print(f"Collecting pooled scores: {name}")
        y_true_all: list[int] = []
        y_proba_all: list[float] = []
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            X_tr, y_tr = _smote(X[train_idx], y[train_idx])
            clf = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(X_tr, y_tr)
                proba = _proba_pos(clf, X[test_idx])
            y_true_all.extend(y[test_idx].tolist())
            y_proba_all.extend(np.asarray(proba, dtype=float).tolist())
        y_true = np.asarray(y_true_all, dtype=int)
        y_proba = np.asarray(y_proba_all, dtype=float)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = float(roc_auc_score(y_true, y_proba))
        out[name] = {"fpr": fpr, "tpr": tpr, "auc": auc}
        print(f"  pooled AUC={auc:.4f}")
    return out


def plot_confusion_grid(metrics: dict, out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()
    for ax, m in zip(axes, metrics["models"]):
        cm = np.asarray(m["confusion_matrix"], dtype=int)
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
            f"{m['model']}\nAcc={m['accuracy_mean']:.3f}  F1m={m['f1_macro_mean']:.3f}"
        )
    fig.suptitle("LOSO confusion matrices (summed over held-out subjects)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(roc_data: dict[str, dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    for name, d in roc_data.items():
        ax.plot(
            d["fpr"],
            d["tpr"],
            color=MODEL_COLORS.get(name, "gray"),
            linewidth=2,
            label=f"{name} (AUC={d['auc']:.3f})",
        )
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Pooled ROC curves (LOSO test predictions concatenated)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_per_subject_accuracy(metrics: dict, out: Path) -> None:
    rows = []
    for m in metrics["models"]:
        for ps in m["per_subject_accuracy"]:
            rows.append(
                {
                    "subject": ps["subject"],
                    "accuracy": ps["accuracy"],
                    "model": m["model"],
                }
            )
    plot_df = pd.DataFrame(rows)
    subjects = sorted(plot_df["subject"].unique())
    plot_df["subject"] = pd.Categorical(
        plot_df["subject"], categories=subjects, ordered=True
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=plot_df, x="subject", y="accuracy", hue="model", ax=ax)
    ax.axhline(
        MAJORITY_BASELINE,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Majority baseline ({MAJORITY_BASELINE:.0%})",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out subject_id")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-subject LOSO accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_f1_comparison(metrics: dict, out: Path) -> None:
    names = [m["model"] for m in metrics["models"]]
    means = [m["f1_macro_mean"] for m in metrics["models"]]
    stds = [m["f1_macro_std"] for m in metrics["models"]]
    colors = [MODEL_COLORS.get(n, "gray") for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, color=colors, capsize=5, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("F1-macro (mean ± std over LOSO folds)")
    ax.set_xlabel("Model")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-macro comparison across models (LOSO)")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="F1 chance (balanced)")
    for bar, mu in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{mu:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_rf_importance(df: pd.DataFrame, out: Path) -> None:
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

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp["feature"], imp["importance"], color="#2E7D32", edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Mean decrease in impurity (RF importance)")
    ax.set_ylabel("Feature")
    ax.set_title("Random Forest feature importance (SMOTE-balanced full EPHNOGRAM set)")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    # Also save CSV for notes / paper
    out_csv = ROOT / "results" / "metrics" / "rf_feature_importance_ephnogram.csv"
    imp.sort_values("importance", ascending=False).to_csv(out_csv, index=False)
    print(f"Saved importance table -> {out_csv}")
    return imp.sort_values("importance", ascending=False)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    metrics = load_metrics()
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded metrics: best={metrics.get('best_model_f1_macro')}")
    print(f"Loaded features: n={len(df)}")

    outs = []

    p1 = FIGURES_DIR / "11_confusion_matrices_grid.png"
    plot_confusion_grid(metrics, p1)
    outs.append(p1)
    print(f"OK {p1.name}")

    roc_data = collect_pooled_scores(df)
    p2 = FIGURES_DIR / "12_roc_curves_pooled.png"
    plot_roc_curves(roc_data, p2)
    outs.append(p2)
    print(f"OK {p2.name}")

    p3 = FIGURES_DIR / "13_per_subject_accuracy.png"
    plot_per_subject_accuracy(metrics, p3)
    outs.append(p3)
    print(f"OK {p3.name}")

    p4 = FIGURES_DIR / "14_f1_macro_comparison.png"
    plot_f1_comparison(metrics, p4)
    outs.append(p4)
    print(f"OK {p4.name}")

    p5 = FIGURES_DIR / "15_rf_feature_importance.png"
    imp = plot_rf_importance(df, p5)
    outs.append(p5)
    print(f"OK {p5.name}")
    print("Top features:\n", imp.head(5).to_string(index=False))

    print("\n=== MODEL VISUALIZATION SUMMARY ===")
    for p in outs:
        print(f"  {p.name} | OK")
    print(f"Generated {len(outs)}/5 plots successfully")


if __name__ == "__main__":
    main()
