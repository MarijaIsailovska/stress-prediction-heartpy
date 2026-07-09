"""
Leave-One-Subject-Out (LOSO) evaluation with SMOTE on train folds only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from config.settings import HRV_FEATURE_COLUMNS, RANDOM_STATE, SMOTE_K_NEIGHBORS

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Accuracy, F1-macro, ROC-AUC, confusion matrix."""
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = SMOTE_K_NEIGHBORS,
) -> tuple[np.ndarray, np.ndarray]:
    """Oversample minority class on the training fold only."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as exc:
        raise ImportError(
            "imbalanced-learn is required for SMOTE. "
            "Install via: pip install imbalanced-learn"
        ) from exc

    # Adapt k if minority class is smaller than k+1
    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    k = min(k_neighbors, max(1, min_count - 1))
    if min_count <= 1:
        logger.warning("Minority class has %d sample(s); skipping SMOTE", min_count)
        return X, y

    smote = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
    return smote.fit_resample(X, y)


def _predict_proba_positive(model, X: np.ndarray) -> np.ndarray | None:
    """Return P(class=1) when available."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            # Assume binary labels {0,1}; use column for class 1 if present
            classes = list(getattr(model, "classes_", [0, 1]))
            if 1 in classes:
                return proba[:, classes.index(1)]
            return proba[:, -1]
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=float)
        # Squash to (0,1) for AUC ranking (monotonic)
        return 1.0 / (1.0 + np.exp(-scores))
    return None


def loso_evaluate(
    df: pd.DataFrame,
    model,
    *,
    feature_cols: list[str] | None = None,
    subject_col: str = "subject_id",
    label_col: str = "label",
    use_smote: bool = True,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Leave-One-Subject-Out cross-validation.

    SMOTE is fit on each training fold only — never on the held-out subject.
    """
    feature_cols = feature_cols or HRV_FEATURE_COLUMNS
    subjects = sorted(df[subject_col].astype(str).unique())
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least 2 subjects")

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    y_proba_all: list[float] = []
    fold_rows: list[dict[str, Any]] = []

    for held_out in subjects:
        train_df = df[df[subject_col].astype(str) != held_out]
        test_df = df[df[subject_col].astype(str) == held_out]
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df[label_col].to_numpy(dtype=int)
        X_test = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df[label_col].to_numpy(dtype=int)

        if use_smote:
            X_train, y_train = _apply_smote(X_train, y_train)

        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = _predict_proba_positive(clf, X_test)

        fold_metrics = compute_metrics(y_test, y_pred, y_proba)
        fold_metrics["subject"] = held_out
        fold_metrics["n_train"] = int(len(y_train))
        fold_metrics["n_test"] = int(len(y_test))
        fold_rows.append(fold_metrics)

        y_true_all.extend(y_test.tolist())
        y_pred_all.extend(np.asarray(y_pred).tolist())
        if y_proba is not None:
            y_proba_all.extend(np.asarray(y_proba).tolist())
        else:
            y_proba_all.extend([float("nan")] * len(y_test))

        logger.info(
            "%s | held-out %s | acc=%.3f f1=%.3f auc=%.3f",
            model_name,
            held_out,
            fold_metrics["accuracy"],
            fold_metrics["f1_macro"],
            fold_metrics["roc_auc"],
        )

    y_true_arr = np.asarray(y_true_all, dtype=int)
    y_pred_arr = np.asarray(y_pred_all, dtype=int)
    y_proba_arr = np.asarray(y_proba_all, dtype=float)
    if np.all(np.isnan(y_proba_arr)):
        overall = compute_metrics(y_true_arr, y_pred_arr, None)
    else:
        overall = compute_metrics(y_true_arr, y_pred_arr, y_proba_arr)

    overall["model"] = model_name
    overall["n_subjects"] = len(subjects)
    overall["n_samples"] = int(len(y_true_arr))
    overall["fold_metrics"] = fold_rows
    return overall


def evaluate_all_models(
    df: pd.DataFrame,
    models: dict[str, Any],
    *,
    use_smote: bool = True,
) -> pd.DataFrame:
    """Run LOSO for every model; return a summary DataFrame."""
    rows = []
    detailed: dict[str, Any] = {}
    for name, model in models.items():
        logger.info("=== LOSO: %s ===", name)
        result = loso_evaluate(df, model, use_smote=use_smote, model_name=name)
        detailed[name] = result
        rows.append(
            {
                "model": name,
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "roc_auc": result["roc_auc"],
                "n_subjects": result["n_subjects"],
                "n_samples": result["n_samples"],
            }
        )
    summary = pd.DataFrame(rows)
    return summary, detailed
