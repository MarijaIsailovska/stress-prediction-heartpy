#!/usr/bin/env python3
"""
03 — Binary stress classification on EPHNOGRAM HRV features.

LOSO cross-validation with SMOTE on each training fold only.
Models: Random Forest, SVM (RBF), KNN, soft Voting Ensemble (RF+SVM+KNN).

Run manually after review:
  python scripts/03_train_evaluate.py
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
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
FEATURES_CSV = ROOT / "data" / "processed" / "ephnogram_features.csv"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR = ROOT / "results" / "figures"
OUT_JSON = METRICS_DIR / "ephnogram_binary_loso.json"

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

# Majority-class baseline (always predict stress) ≈ 465/607 ≈ 76.6%
MAJORITY_BASELINE_ACC = 0.77

COLOR_REST = "#2196F3"
COLOR_STRESS = "#F44336"


def build_models() -> dict:
    """Four study models with specified hyperparameters."""
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    svm = Pipeline(
        steps=[
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
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    # Fresh estimators for the ensemble
    rf_e = RandomForestClassifier(
        n_estimators=200,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    svm_e = Pipeline(
        steps=[
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
    knn_e = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=5,
                    weights="distance",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    ensemble = VotingClassifier(
        estimators=[
            ("rf", rf_e),
            ("svm", svm_e),
            ("knn", knn_e),
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


def _apply_smote(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SMOTE on training fold only; adapt k if minority class is small."""
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    if min_count <= 1:
        return X, y
    k = min(SMOTE_K, max(1, min_count - 1))
    smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    return smote.fit_resample(X, y)


def _predict_proba_positive(model, X: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", [0, 1]))
    if 1 in classes:
        return proba[:, classes.index(1)]
    return proba[:, -1]


def loso_evaluate(df: pd.DataFrame, model, model_name: str) -> dict:
    """
    Leave-One-Subject-Out CV.

    Per fold: SMOTE(train) → fit → predict(test). Never SMOTE the test fold.
    SVM/KNN/Ensemble include StandardScaler inside their pipelines (fit on train).

    Per-fold ROC-AUC is undefined when the held-out subject has only one class
    (common with per-recording subject_ids). We still report fold-wise nan AUC
    and a **pooled** ROC-AUC over all concatenated test predictions.
    """
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].to_numpy(dtype=int)
    groups = df[GROUP_COL].astype(str).to_numpy()

    logo = LeaveOneGroupOut()
    fold_rows: list[dict] = []
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_proba_all: list[float] = []
    per_subject: list[dict] = []

    for train_idx, test_idx in logo.split(X, y, groups):
        held_out = str(groups[test_idx][0])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        X_train_bal, y_train_bal = _apply_smote(X_train, y_train)

        clf = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_train_bal, y_train_bal)
            y_pred = clf.predict(X_test)
            y_proba = _predict_proba_positive(clf, X_test)

        # STEP 1 — debug: class composition of held-out fold
        proba_shape = None if y_proba is None else tuple(y_proba.shape)
        print(
            f"  Subject {held_out}: y_test classes={np.unique(y_test)}, "
            f"n_test={len(y_test)}, proba_shape={proba_shape}"
        )

        acc = float(accuracy_score(y_test, y_pred))
        f1m = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        f1w = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

        # STEP 3 — fold AUC only if both classes present in y_test
        try:
            if len(np.unique(y_test)) < 2:
                auc = float("nan")
            elif y_proba is None:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            auc = float("nan")

        fold_rows.append(
            {
                "subject": held_out,
                "n_train": int(len(y_train_bal)),
                "n_test": int(len(y_test)),
                "n_classes_test": int(len(np.unique(y_test))),
                "accuracy": acc,
                "f1_macro": f1m,
                "f1_weighted": f1w,
                "roc_auc": auc,
            }
        )
        per_subject.append({"subject": held_out, "accuracy": acc, "n_test": int(len(y_test))})
        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(np.asarray(y_pred).tolist())
        if y_proba is not None:
            y_proba_all.extend(np.asarray(y_proba, dtype=float).tolist())
        else:
            y_proba_all.extend([float("nan")] * len(y_test))

    metrics = ["accuracy", "f1_macro", "f1_weighted", "roc_auc"]
    summary = {"model": model_name, "n_folds": len(fold_rows)}
    for m in metrics:
        vals = np.asarray([r[m] for r in fold_rows], dtype=float)
        if np.all(np.isnan(vals)):
            summary[f"{m}_mean"] = float("nan")
            summary[f"{m}_std"] = float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                summary[f"{m}_mean"] = float(np.nanmean(vals))
                summary[f"{m}_std"] = float(np.nanstd(vals, ddof=0))

    # Pooled ROC-AUC (valid when each fold is single-class but overall has both)
    y_true_arr = np.asarray(y_true_all, dtype=int)
    y_proba_arr = np.asarray(y_proba_all, dtype=float)
    try:
        if len(np.unique(y_true_arr)) < 2 or np.all(np.isnan(y_proba_arr)):
            pooled_auc = float("nan")
        else:
            pooled_auc = float(roc_auc_score(y_true_arr, y_proba_arr))
    except ValueError:
        pooled_auc = float("nan")
    summary["roc_auc_pooled"] = pooled_auc
    # Table display: use pooled AUC when fold-wise mean is nan
    if np.isnan(summary["roc_auc_mean"]) and not np.isnan(pooled_auc):
        summary["roc_auc_mean"] = pooled_auc
        summary["roc_auc_std"] = float("nan")  # pooled scalar, no fold std
        summary["roc_auc_note"] = (
            "roc_auc_mean is pooled across folds (per-fold AUC undefined: "
            "each held-out subject has a single class)"
        )

    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    summary["confusion_matrix"] = cm.tolist()
    summary["folds"] = fold_rows
    summary["per_subject_accuracy"] = per_subject
    return summary


def print_results_table(results: list[dict]) -> None:
    print("\nModel          | Accuracy      | F1-macro      | F1-weighted   | ROC-AUC (pooled)")
    print("-" * 82)
    for r in results:
        auc_mean = r["roc_auc_mean"]
        auc_std = r["roc_auc_std"]
        if np.isnan(auc_std):
            auc_str = f"{auc_mean:.3f} (pooled)" if not np.isnan(auc_mean) else "nan"
        else:
            auc_str = f"{auc_mean:.3f} ± {auc_std:.3f}"
        print(
            f"{r['model']:<14} | "
            f"{r['accuracy_mean']:.3f} ± {r['accuracy_std']:.3f} | "
            f"{r['f1_macro_mean']:.3f} ± {r['f1_macro_std']:.3f} | "
            f"{r['f1_weighted_mean']:.3f} ± {r['f1_weighted_std']:.3f} | "
            f"{auc_str}"
        )
    print(
        f"\nMajority-class baseline (always predict stress) ≈ "
        f"{MAJORITY_BASELINE_ACC:.0%} accuracy — models should beat this."
    )
    if any(r.get("roc_auc_note") for r in results):
        print(
            "Note: Per-fold ROC-AUC is undefined when each subject_id has only "
            "one label (current per-recording SUBJECT_MAP). Table ROC-AUC is "
            "pooled over all test predictions."
        )


def plot_confusion(cm: np.ndarray, model_name: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REST", "STRESS"],
        yticklabels=["REST", "STRESS"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"LOSO confusion matrix — {model_name}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_subject_accuracy(results: list[dict], out_path: Path) -> None:
    """Grouped bar chart: accuracy per held-out subject for each model."""
    rows = []
    for r in results:
        for ps in r["per_subject_accuracy"]:
            rows.append(
                {
                    "subject": ps["subject"],
                    "accuracy": ps["accuracy"],
                    "model": r["model"],
                }
            )
    plot_df = pd.DataFrame(rows)
    # Stable subject order
    subjects = sorted(plot_df["subject"].unique())
    plot_df["subject"] = pd.Categorical(plot_df["subject"], categories=subjects, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=plot_df,
        x="subject",
        y="accuracy",
        hue="model",
        ax=ax,
    )
    ax.axhline(
        MAJORITY_BASELINE_ACC,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Majority baseline ({MAJORITY_BASELINE_ACC:.0%})",
    )
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Held-out subject_id (LOSO fold)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-subject LOSO accuracy (hardest subjects = lowest bars)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    n_subj = df[GROUP_COL].nunique()
    print(f"Loaded {len(df)} windows | rest={n_rest} stress={n_stress} | subjects={n_subj}")
    print(f"Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"Majority baseline ≈ {n_stress / len(df):.1%} (always predict stress)\n")

    # STEP 2 — which subjects have only one class? (explains nan per-fold AUC)
    print("=== Subject class inventory (LOSO groups) ===")
    n_single = 0
    for subj in sorted(df[GROUP_COL].astype(str).unique()):
        sub = df[df[GROUP_COL].astype(str) == subj]
        labels = sorted(sub[LABEL_COL].unique().tolist())
        print(f"Subject {subj}: classes={labels}, n={len(sub)}")
        if len(labels) < 2:
            n_single += 1
    print(
        f"Subjects with a single class: {n_single}/{n_subj} "
        f"(per-fold ROC-AUC undefined for these folds)\n"
    )

    models = build_models()
    results: list[dict] = []
    for name, model in models.items():
        print(f"=== LOSO: {name} ===")
        res = loso_evaluate(df, model, name)
        results.append(res)
        print(
            f"  Acc={res['accuracy_mean']:.3f}±{res['accuracy_std']:.3f}  "
            f"F1m={res['f1_macro_mean']:.3f}±{res['f1_macro_std']:.3f}  "
            f"AUC={res['roc_auc_mean']:.3f}±{res['roc_auc_std']:.3f}"
        )

    print_results_table(results)

    # Best model by mean F1-macro
    best = max(results, key=lambda r: r["f1_macro_mean"])
    best_name = best["model"]
    safe_name = best_name.replace(" ", "_").lower()
    cm_path = FIGURES_DIR / f"confusion_matrix_{safe_name}.png"
    plot_confusion(np.asarray(best["confusion_matrix"]), best_name, cm_path)
    print(f"\nBest model by F1-macro: {best_name}")
    print(f"Saved confusion matrix -> {cm_path}")

    subj_path = FIGURES_DIR / "per_subject_accuracy.png"
    plot_per_subject_accuracy(results, subj_path)
    print(f"Saved per-subject accuracy -> {subj_path}")

    # JSON (convert numpy-friendly)
    payload = {
        "dataset": "ephnogram",
        "n_samples": int(len(df)),
        "n_rest": n_rest,
        "n_stress": n_stress,
        "n_subjects": int(n_subj),
        "features": FEATURE_COLS,
        "majority_baseline_accuracy": float(n_stress / len(df)),
        "best_model_f1_macro": best_name,
        "models": results,
    }
    with OUT_JSON.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved metrics -> {OUT_JSON}")


if __name__ == "__main__":
    main()
