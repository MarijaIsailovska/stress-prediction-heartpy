#!/usr/bin/env python3
"""
04 — EDA visualizations for EPHNOGRAM HRV features.

Loads data/processed/ephnogram_features.csv and writes publication-quality
figures (DPI=300) to results/figures/.

Does not train models. Run manually after review:
  python scripts/04_eda_visualizations.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths & style
# ---------------------------------------------------------------------------
FEATURES_CSV = ROOT / "data" / "processed" / "ephnogram_features.csv"
FIG_DIR = ROOT / "results" / "figures"

COLOR_REST = "#2196F3"
COLOR_STRESS = "#F44336"
PALETTE = {0: COLOR_REST, 1: COLOR_STRESS}
LABEL_NAMES = {0: "REST", 1: "STRESS"}
DPI = 300

HRV_FEATURES = [
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

FEATURE_TITLES = {
    "bpm": "BPM",
    "ibi": "IBI (ms)",
    "sdnn": "SDNN (ms)",
    "rmssd": "RMSSD (ms)",
    "sdsd": "SDSD (ms)",
    "pnn20": "pNN20",
    "pnn50": "pNN50",
    "hr_mad": "HR_MAD (ms)",
    "sd1": "SD1 (ms)",
    "sd2": "SD2 (ms)",
    "breathing_rate": "Breathing rate (Hz)",
}


def load_features() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"Missing features CSV: {FEATURES_CSV}")
    df = pd.read_csv(FEATURES_CSV)
    missing = [c for c in HRV_FEATURES + ["label", "subject_id", "recording_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    return df


def _mw_stars(x: np.ndarray, y: np.ndarray) -> str:
    """Mann–Whitney U two-sided; return significance stars."""
    if len(x) < 2 or len(y) < 2:
        return "ns"
    try:
        _stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
    except ValueError:
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _confidence_ellipse(ax, x, y, n_std=1.96, **kwargs):
    """95% confidence ellipse from covariance eigenvalues (n_std≈1.96)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return None
    cov = np.cov(x, y)
    if cov.shape != (2, 2) or not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 0.0)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=theta,
        **kwargs,
    )
    ax.add_patch(ell)
    return ell


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_01_hrv_boxplots(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    axes = axes.ravel()
    plot_df = df.copy()
    plot_df["class"] = plot_df["label"].map(LABEL_NAMES)

    for i, feat in enumerate(HRV_FEATURES):
        ax = axes[i]
        sns.boxplot(
            data=plot_df,
            x="class",
            y=feat,
            hue="class",
            palette={"REST": COLOR_REST, "STRESS": COLOR_STRESS},
            legend=False,
            ax=ax,
            order=["REST", "STRESS"],
        )
        rest = df.loc[df["label"] == 0, feat].dropna().to_numpy()
        stress = df.loc[df["label"] == 1, feat].dropna().to_numpy()
        stars = _mw_stars(rest, stress)
        y_max = np.nanmax(df[feat].to_numpy())
        y_min = np.nanmin(df[feat].to_numpy())
        ax.text(
            0.5,
            y_max + 0.05 * (y_max - y_min + 1e-9),
            stars,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
        ax.set_title(FEATURE_TITLES[feat])
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_TITLES[feat])

    axes[-1].axis("off")
    fig.suptitle(
        "HRV feature distributions: REST vs STRESS\n"
        "(Mann–Whitney U: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_02_correlation_heatmap(df: pd.DataFrame, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, lab, title in [
        (axes[0], 0, "REST"),
        (axes[1], 1, "STRESS"),
    ]:
        sub = df.loc[df["label"] == lab, HRV_FEATURES]
        corr = sub.corr(method="pearson")
        sns.heatmap(
            corr,
            ax=ax,
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            annot=True,
            fmt=".2f",
            annot_kws={"size": 7},
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
        )
        ax.set_title(f"{title} — feature correlations")
        ax.set_xticklabels([FEATURE_TITLES[c].split()[0] for c in HRV_FEATURES], rotation=45, ha="right")
        ax.set_yticklabels([FEATURE_TITLES[c].split()[0] for c in HRV_FEATURES], rotation=0)

    fig.suptitle(
        "Do HRV feature correlations change under physical stress?\n"
        "(multicollinearity context for feature selection)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_03_poincare(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for lab, color, name in [
        (0, COLOR_REST, "REST"),
        (1, COLOR_STRESS, "STRESS"),
    ]:
        sub = df[df["label"] == lab]
        ax.scatter(
            sub["sd1"],
            sub["sd2"],
            c=color,
            alpha=0.45,
            s=28,
            label=name,
            edgecolors="none",
        )
        _confidence_ellipse(
            ax,
            sub["sd1"],
            sub["sd2"],
            n_std=1.96,
            facecolor="none",
            edgecolor=color,
            linewidth=2.0,
            label=f"{name} 95% ellipse",
        )

    ax.set_xlabel("SD1 (ms) — short-term variability")
    ax.set_ylabel("SD2 (ms) — long-term variability")
    ax.set_title("Poincaré summary: SD1 vs SD2 by class")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_04_sd1_sd2_ratio(df: pd.DataFrame, out: Path) -> None:
    plot_df = df.copy()
    plot_df["sd1_sd2"] = plot_df["sd1"] / plot_df["sd2"].replace(0, np.nan)
    plot_df["class"] = plot_df["label"].map(LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(
        data=plot_df.dropna(subset=["sd1_sd2"]),
        x="class",
        y="sd1_sd2",
        hue="class",
        palette={"REST": COLOR_REST, "STRESS": COLOR_STRESS},
        order=["REST", "STRESS"],
        inner="box",
        legend=False,
        ax=ax,
        cut=0,
    )
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5, label="Balance (SD1/SD2 = 1)")
    ax.set_ylabel("SD1 / SD2 ratio")
    ax.set_xlabel("Class")
    ax.set_title("SD1/SD2 ratio (parasympathetic / sympathetic proxy)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_05_rmssd_bpm_scatter(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    for lab, color, name in [
        (0, COLOR_REST, "REST"),
        (1, COLOR_STRESS, "STRESS"),
    ]:
        sub = df[df["label"] == lab]
        ax.scatter(
            sub["bpm"],
            sub["rmssd"],
            c=color,
            alpha=0.5,
            s=28,
            label=name,
            edgecolors="none",
        )

    ax.axvline(100, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(50, color="black", linestyle="--", linewidth=1.2)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(
        (xlim[0] + 100) / 2,
        (50 + ylim[1]) / 2,
        "REST zone\n(low BPM, high RMSSD)",
        ha="center",
        va="center",
        fontsize=9,
        color=COLOR_REST,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor=COLOR_REST),
    )
    ax.text(
        (100 + xlim[1]) / 2,
        (ylim[0] + 50) / 2,
        "STRESS zone\n(high BPM, low RMSSD)",
        ha="center",
        va="center",
        fontsize=9,
        color=COLOR_STRESS,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor=COLOR_STRESS),
    )

    ax.set_xlabel("BPM (heart rate)")
    ax.set_ylabel("RMSSD (ms)")
    ax.set_title("RMSSD vs BPM — rest vs physical stress")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_06_pca_biplot(df: pd.DataFrame, out: Path) -> None:
    X = df[HRV_FEATURES].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    scores = pca.fit_transform(Xs)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    fig, ax = plt.subplots(figsize=(9, 7))
    for lab, color, name in [
        (0, COLOR_REST, "REST"),
        (1, COLOR_STRESS, "STRESS"),
    ]:
        mask = y == lab
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            c=color,
            alpha=0.5,
            s=28,
            label=name,
            edgecolors="none",
        )

    # Scale arrows for visibility
    scale = 3.0
    for i, feat in enumerate(HRV_FEATURES):
        ax.arrow(
            0,
            0,
            loadings[i, 0] * scale,
            loadings[i, 1] * scale,
            color="black",
            alpha=0.75,
            head_width=0.08,
            length_includes_head=True,
        )
        ax.text(
            loadings[i, 0] * scale * 1.12,
            loadings[i, 1] * scale * 1.12,
            feat,
            fontsize=8,
            ha="center",
            va="center",
        )

    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1f}% variance)")
    ax.set_title("PCA biplot — are REST/STRESS separable in HRV space?")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axvline(0, color="gray", lw=0.8)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_07_subject_profiles(df: pd.DataFrame, out: Path) -> None:
    prof = (
        df.groupby(["subject_id", "label"], as_index=False)
        .agg(mean_bpm=("bpm", "mean"), mean_rmssd=("rmssd", "mean"))
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    for lab, color, name in [
        (0, COLOR_REST, "REST"),
        (1, COLOR_STRESS, "STRESS"),
    ]:
        sub = prof[prof["label"] == lab]
        ax.scatter(
            sub["mean_bpm"],
            sub["mean_rmssd"],
            c=color,
            s=70,
            label=name,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                str(row["subject_id"]),
                (row["mean_bpm"], row["mean_rmssd"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=7,
            )

    ax.set_xlabel("Mean BPM (per subject)")
    ax.set_ylabel("Mean RMSSD (ms, per subject)")
    ax.set_title("Per-subject HRV profiles (context for LOSO CV)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_08_breathing_rate(df: pd.DataFrame, out: Path) -> None:
    plot_df = df.copy()
    plot_df["breaths_per_min"] = plot_df["breathing_rate"] * 60.0
    plot_df["class"] = plot_df["label"].map(LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.violinplot(
        data=plot_df,
        x="class",
        y="breaths_per_min",
        hue="class",
        palette={"REST": COLOR_REST, "STRESS": COLOR_STRESS},
        order=["REST", "STRESS"],
        inner="box",
        legend=False,
        ax=ax,
        cut=0,
    )
    ax.axhline(12, color="gray", linestyle="--", linewidth=1.2, label="Normal lower (12 /min)")
    ax.axhline(20, color="gray", linestyle=":", linewidth=1.2, label="Normal upper (20 /min)")
    ax.set_ylabel("Breathing rate (breaths / min)")
    ax.set_xlabel("Class")
    ax.set_title("Estimated breathing rate by class (HeartPy)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_09_class_distribution(df: pd.DataFrame, out: Path) -> None:
    counts = (
        df.groupby(["recording_id", "label"], as_index=False)
        .size()
        .rename(columns={"size": "n_windows"})
        .sort_values("recording_id")
    )
    colors = [COLOR_REST if lab == 0 else COLOR_STRESS for lab in counts["label"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(counts["recording_id"], counts["n_windows"], color=colors)
    axes[0].set_xlabel("recording_id")
    axes[0].set_ylabel("Number of windows")
    axes[0].set_title("Windows kept per recording")
    axes[0].tick_params(axis="x", rotation=90, labelsize=7)
    # Manual legend
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[
            Patch(facecolor=COLOR_REST, label="REST"),
            Patch(facecolor=COLOR_STRESS, label="STRESS"),
        ],
        loc="best",
    )

    n_rest = int((df["label"] == 0).sum())
    n_stress = int((df["label"] == 1).sum())
    total = n_rest + n_stress
    axes[1].pie(
        [n_rest, n_stress],
        labels=[
            f"REST\n{n_rest} ({100 * n_rest / total:.1f}%)",
            f"STRESS\n{n_stress} ({100 * n_stress / total:.1f}%)",
        ],
        colors=[COLOR_REST, COLOR_STRESS],
        startangle=90,
        wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("Overall class distribution")
    axes[1].legend(["REST", "STRESS"], loc="best")

    fig.suptitle("Class imbalance overview (justifies SMOTE on train folds)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_10_pairplot(df: pd.DataFrame, out: Path) -> None:
    cols = ["bpm", "rmssd", "sd1", "sd2"]
    plot_df = df[cols + ["label"]].copy()
    plot_df["class"] = plot_df["label"].map(LABEL_NAMES)

    g = sns.pairplot(
        plot_df,
        vars=cols,
        hue="class",
        palette={"REST": COLOR_REST, "STRESS": COLOR_STRESS},
        diag_kind="kde",
        plot_kws={"alpha": 0.45, "s": 18, "edgecolor": "none"},
        diag_kws={"fill": True, "alpha": 0.4},
        height=2.2,
    )
    g.figure.suptitle(
        "Pairwise HRV features (bpm, RMSSD, SD1, SD2) — class separability",
        y=1.02,
        fontsize=12,
    )
    g.figure.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(g.figure)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
PLOT_SPECS = [
    ("01_hrv_boxplots.png", plot_01_hrv_boxplots),
    ("02_correlation_heatmap.png", plot_02_correlation_heatmap),
    ("03_poincare_plot.png", plot_03_poincare),
    ("04_sd1_sd2_ratio.png", plot_04_sd1_sd2_ratio),
    ("05_rmssd_bpm_scatter.png", plot_05_rmssd_bpm_scatter),
    ("06_pca_biplot.png", plot_06_pca_biplot),
    ("07_subject_profiles.png", plot_07_subject_profiles),
    ("08_breathing_rate.png", plot_08_breathing_rate),
    ("09_class_distribution.png", plot_09_class_distribution),
    ("10_pairplot.png", plot_10_pairplot),
]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    df = load_features()
    print(f"Loaded {len(df)} windows from {FEATURES_CSV}")
    print(f"Figures -> {FIG_DIR}\n")

    results: list[tuple[str, str]] = []
    for filename, fn in PLOT_SPECS:
        out = FIG_DIR / filename
        try:
            fn(df, out)
            status = "OK"
            print(f"  OK  {filename}")
        except Exception as exc:  # noqa: BLE001
            status = f"FAILED: {exc}"
            print(f"  FAIL {filename}: {exc}")
            traceback.print_exc()
        results.append((filename, status))

    print("\n=== EDA VISUALIZATION SUMMARY ===")
    n_ok = 0
    for filename, status in results:
        print(f"  {filename} | {status}")
        if status == "OK":
            n_ok += 1
    print(f'Generated {n_ok}/10 plots successfully')


if __name__ == "__main__":
    main()
