"""
Cross-sensor generalization: train on EPHNOGRAM ECG → test on Wrist PPG.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from config.settings import HRV_FEATURE_COLUMNS, SMOTE_K_NEIGHBORS
from src.models.evaluation import _apply_smote, _predict_proba_positive, compute_metrics

logger = logging.getLogger(__name__)


def cross_sensor_evaluate(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model,
    *,
    feature_cols: list[str] | None = None,
    label_col: str = "label",
    use_smote: bool = True,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Train on one sensor dataset (typically EPHNOGRAM ECG) and evaluate
    on another (typically Wrist PPG), using the shared HRV feature space.

    Addresses the research question: can a model trained on clinical chest
    ECG generalize to wrist PPG as used in soldier/firefighter wearables?
    """
    feature_cols = feature_cols or HRV_FEATURE_COLUMNS

    missing_train = [c for c in feature_cols if c not in train_df.columns]
    missing_test = [c for c in feature_cols if c not in test_df.columns]
    if missing_train or missing_test:
        raise ValueError(
            f"Missing feature columns — train: {missing_train}, test: {missing_test}"
        )

    X_train = train_df[feature_cols].to_numpy(dtype=float)
    y_train = train_df[label_col].to_numpy(dtype=int)
    X_test = test_df[feature_cols].to_numpy(dtype=float)
    y_test = test_df[label_col].to_numpy(dtype=int)

    if use_smote:
        X_train, y_train = _apply_smote(
            X_train, y_train, k_neighbors=SMOTE_K_NEIGHBORS
        )

    clf = clone(model)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = _predict_proba_positive(clf, X_test)

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["model"] = model_name
    metrics["n_train"] = int(len(y_train))
    metrics["n_test"] = int(len(y_test))
    metrics["train_sensor"] = (
        str(train_df["sensor"].iloc[0]) if "sensor" in train_df.columns else "train"
    )
    metrics["test_sensor"] = (
        str(test_df["sensor"].iloc[0]) if "sensor" in test_df.columns else "test"
    )
    metrics["y_true"] = y_test.tolist()
    metrics["y_pred"] = np.asarray(y_pred).tolist()

    logger.info(
        "Cross-sensor %s | train=%s (%d) → test=%s (%d) | acc=%.3f f1=%.3f auc=%.3f",
        model_name,
        metrics["train_sensor"],
        metrics["n_train"],
        metrics["test_sensor"],
        metrics["n_test"],
        metrics["accuracy"],
        metrics["f1_macro"],
        metrics["roc_auc"],
    )
    return metrics


def cross_sensor_all_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models: dict[str, Any],
    *,
    use_smote: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run cross-sensor evaluation for every model."""
    rows = []
    detailed: dict[str, Any] = {}
    for name, model in models.items():
        result = cross_sensor_evaluate(
            train_df,
            test_df,
            model,
            use_smote=use_smote,
            model_name=name,
        )
        detailed[name] = result
        rows.append(
            {
                "model": name,
                "accuracy": result["accuracy"],
                "f1_macro": result["f1_macro"],
                "roc_auc": result["roc_auc"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "train_sensor": result["train_sensor"],
                "test_sensor": result["test_sensor"],
            }
        )
    return pd.DataFrame(rows), detailed
