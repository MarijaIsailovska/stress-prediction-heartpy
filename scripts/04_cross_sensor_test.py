#!/usr/bin/env python3
"""
04 — Cross-sensor generalization experiment.

Train on EPHNOGRAM (ECG chest) → test on Wrist PPG.
Addresses: can clinical-ECG-trained models generalize to smartwatch PPG?
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import clone

from config.settings import (
    EPHNOGRAM_FEATURES_CSV,
    FIGURES_DIR,
    HRV_FEATURE_COLUMNS,
    METRICS_DIR,
    MODELS_DIR,
    WRIST_FEATURES_CSV,
)
from src.models.classifiers import build_classifiers
from src.models.cross_sensor import cross_sensor_all_models
from src.models.evaluation import _apply_smote
from src.utils.io import ensure_dirs, load_dataframe, save_json, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("04_cross_sensor_test")


def main() -> None:
    ensure_dirs()

    if not EPHNOGRAM_FEATURES_CSV.exists() or not WRIST_FEATURES_CSV.exists():
        logger.error(
            "Need both feature CSVs. Run scripts 01 and 02 first.\n  %s\n  %s",
            EPHNOGRAM_FEATURES_CSV,
            WRIST_FEATURES_CSV,
        )
        sys.exit(1)

    train_df = load_dataframe(EPHNOGRAM_FEATURES_CSV)
    test_df = load_dataframe(WRIST_FEATURES_CSV)
    logger.info(
        "Train ECG: %d samples | Test PPG: %d samples",
        len(train_df),
        len(test_df),
    )

    models = build_classifiers(include_extra=True)
    summary, detailed = cross_sensor_all_models(
        train_df, test_df, models, use_smote=True
    )

    summary.to_csv(METRICS_DIR / "cross_sensor_summary.csv", index=False)
    save_json(detailed, METRICS_DIR / "cross_sensor_detailed.json")
    logger.info("Cross-sensor summary:\n%s", summary.to_string(index=False))

    best = summary.sort_values("f1_macro", ascending=False).iloc[0]
    best_name = best["model"]
    cm = detailed[best_name]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        np.asarray(cm),
        annot=True,
        fmt="d",
        cmap="Oranges",
        xticklabels=["Rest", "Stress"],
        yticklabels=["Rest", "Stress"],
        ax=ax,
    )
    ax.set_title(f"Cross-sensor (ECG→PPG) — {best_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cm_cross_sensor_{best_name}.png", dpi=150)
    plt.close(fig)

    X = train_df[HRV_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train_df["label"].to_numpy(dtype=int)
    X_bal, y_bal = _apply_smote(X, y)
    clf = clone(models[best_name])
    clf.fit(X_bal, y_bal)
    save_model(clf, MODELS_DIR / f"cross_sensor_best_{best_name}.joblib")
    logger.info("Saved best cross-sensor model: %s", best_name)


if __name__ == "__main__":
    main()
