#!/usr/bin/env python3
"""
03 — Train & evaluate classifiers with LOSO + SMOTE (train fold only).

RQ1: HRV features classify physical stress vs rest from ECG (binary).
RQ2: Compare ECG (EPHNOGRAM) vs PPG (Wrist) classification performance.
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
import pandas as pd
import seaborn as sns

from config.settings import (
    EPHNOGRAM_FEATURES_CSV,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    WRIST_FEATURES_CSV,
)
from src.models.classifiers import build_classifiers
from src.models.evaluation import evaluate_all_models
from src.utils.io import ensure_dirs, load_dataframe, save_json, save_model
from sklearn.base import clone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("03_train_evaluate")


def _plot_confusion(cm: list[list[int]], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(
        np.asarray(cm),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Rest", "Stress"],
        yticklabels=["Rest", "Stress"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_dataset(name: str, csv_path: Path) -> None:
    if not csv_path.exists():
        logger.warning("Skipping %s — missing %s", name, csv_path)
        return

    df = load_dataframe(csv_path)
    logger.info("%s: %d samples, %d subjects, label counts:\n%s",
                name, len(df), df["subject_id"].nunique(), df["label"].value_counts().to_string())

    models = build_classifiers(include_extra=True)
    summary, detailed = evaluate_all_models(df, models, use_smote=True)

    out_csv = METRICS_DIR / f"loso_{name}_summary.csv"
    summary.to_csv(out_csv, index=False)
    save_json(detailed, METRICS_DIR / f"loso_{name}_detailed.json")
    logger.info("Summary (%s):\n%s", name, summary.to_string(index=False))

    # Confusion matrices for each model
    for model_name, result in detailed.items():
        cm = result.get("confusion_matrix")
        if cm is not None:
            _plot_confusion(
                cm,
                f"LOSO — {name} — {model_name}",
                FIGURES_DIR / f"cm_loso_{name}_{model_name}.png",
            )

    # Fit final models on full data (for later SHAP / cross-sensor) and save RF
    from config.settings import HRV_FEATURE_COLUMNS
    from src.models.evaluation import _apply_smote

    X = df[HRV_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    X_bal, y_bal = _apply_smote(X, y)
    rf = clone(models["RandomForest"])
    rf.fit(X_bal, y_bal)
    save_model(rf, MODELS_DIR / f"rf_{name}_full.joblib")


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")
    run_dataset("ephnogram", EPHNOGRAM_FEATURES_CSV)
    run_dataset("wrist", WRIST_FEATURES_CSV)

    # Side-by-side RQ2 comparison if both exist
    eph = METRICS_DIR / "loso_ephnogram_summary.csv"
    wri = METRICS_DIR / "loso_wrist_summary.csv"
    if eph.exists() and wri.exists():
        a = pd.read_csv(eph).assign(dataset="ECG_EPHNOGRAM")
        b = pd.read_csv(wri).assign(dataset="PPG_Wrist")
        cmp = pd.concat([a, b], ignore_index=True)
        cmp.to_csv(METRICS_DIR / "rq2_ecg_vs_ppg.csv", index=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=cmp, x="model", y="f1_macro", hue="dataset", ax=ax)
        ax.set_title("RQ2: ECG (chest) vs PPG (wrist) — F1-macro (LOSO)")
        ax.set_ylabel("F1-macro")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "rq2_ecg_vs_ppg_f1.png", dpi=150)
        plt.close(fig)
        logger.info("RQ2 comparison saved.")


if __name__ == "__main__":
    main()
