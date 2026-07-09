#!/usr/bin/env python3
"""
05 — Feature importance (Random Forest) + SHAP explanations.

RQ3: Which HRV features are strongest predictors regardless of sensor?
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
from sklearn.ensemble import RandomForestClassifier

from config.settings import (
    EPHNOGRAM_FEATURES_CSV,
    FIGURES_DIR,
    HRV_FEATURE_COLUMNS,
    METRICS_DIR,
    RANDOM_STATE,
    RF_PARAMS,
    WRIST_FEATURES_CSV,
)
from src.models.evaluation import _apply_smote
from src.utils.io import ensure_dirs, load_dataframe, save_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("05_feature_importance")


def load_combined() -> pd.DataFrame:
    frames = []
    for path in (EPHNOGRAM_FEATURES_CSV, WRIST_FEATURES_CSV):
        if path.exists():
            frames.append(load_dataframe(path))
        else:
            logger.warning("Missing %s", path)
    if not frames:
        raise FileNotFoundError("No processed feature CSVs found. Run scripts 01/02.")
    return pd.concat(frames, ignore_index=True)


def rf_importance(df: pd.DataFrame, tag: str) -> pd.DataFrame:
    X = df[HRV_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    X_bal, y_bal = _apply_smote(X, y)

    rf = RandomForestClassifier(**RF_PARAMS)
    rf.fit(X_bal, y_bal)

    imp = pd.DataFrame(
        {
            "feature": HRV_FEATURE_COLUMNS,
            "importance": rf.feature_importances_,
            "dataset": tag,
        }
    ).sort_values("importance", ascending=False)

    out_csv = METRICS_DIR / f"feature_importance_rf_{tag}.csv"
    imp.to_csv(out_csv, index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=imp, y="feature", x="importance", ax=ax, color="#2c5f7c")
    ax.set_title(f"RF feature importance — {tag}")
    ax.set_xlabel("Mean decrease in impurity")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"feature_importance_rf_{tag}.png", dpi=150)
    plt.close(fig)

    return imp, rf, X, y


def shap_summary(rf: RandomForestClassifier, X: np.ndarray, tag: str) -> None:
    try:
        import shap
    except ImportError:
        logger.warning("shap not installed; skipping SHAP plots for %s", tag)
        return

    # Subsample for speed on large matrices
    rng = np.random.default_rng(RANDOM_STATE)
    n = min(500, len(X))
    idx = rng.choice(len(X), size=n, replace=False)
    X_s = X[idx]

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X_s)

    # Binary RF: shap_values may be list [class0, class1] or ndarray
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
        if sv.ndim == 3:
            sv = sv[:, :, 1]

    mean_abs = np.abs(sv).mean(axis=0)
    shap_rank = pd.DataFrame(
        {"feature": HRV_FEATURE_COLUMNS, "mean_abs_shap": mean_abs, "dataset": tag}
    ).sort_values("mean_abs_shap", ascending=False)
    shap_rank.to_csv(METRICS_DIR / f"shap_importance_{tag}.csv", index=False)

    plt.figure(figsize=(8, 5))
    shap.summary_plot(
        sv,
        X_s,
        feature_names=HRV_FEATURE_COLUMNS,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_summary_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    shap.summary_plot(
        sv,
        X_s,
        feature_names=HRV_FEATURE_COLUMNS,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"shap_bar_{tag}.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP plots saved for %s", tag)


def main() -> None:
    ensure_dirs()
    sns.set_theme(style="whitegrid")

    results = {}

    # Per-sensor and combined (RQ3: regardless of sensor)
    datasets: dict[str, pd.DataFrame] = {}
    if EPHNOGRAM_FEATURES_CSV.exists():
        datasets["ephnogram"] = load_dataframe(EPHNOGRAM_FEATURES_CSV)
    if WRIST_FEATURES_CSV.exists():
        datasets["wrist"] = load_dataframe(WRIST_FEATURES_CSV)
    if len(datasets) >= 1:
        datasets["combined"] = load_combined()

    if not datasets:
        logger.error("No data. Run scripts 01 and/or 02 first.")
        sys.exit(1)

    all_imp = []
    for tag, df in datasets.items():
        logger.info("=== Feature importance: %s (%d samples) ===", tag, len(df))
        imp, rf, X, y = rf_importance(df, tag)
        all_imp.append(imp)
        shap_summary(rf, X, tag)
        results[tag] = {
            "top_features": imp.head(5)["feature"].tolist(),
            "n_samples": int(len(df)),
        }

    merged = pd.concat(all_imp, ignore_index=True)
    merged.to_csv(METRICS_DIR / "feature_importance_all.csv", index=False)

    # Comparison bar chart across datasets
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(
        data=merged,
        y="feature",
        x="importance",
        hue="dataset",
        ax=ax,
        order=HRV_FEATURE_COLUMNS,
    )
    ax.set_title("RQ3: HRV feature importance across sensors")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "rq3_feature_importance_comparison.png", dpi=150)
    plt.close(fig)

    save_json(results, METRICS_DIR / "rq3_top_features.json")
    logger.info("RQ3 top features: %s", results)


if __name__ == "__main__":
    main()
