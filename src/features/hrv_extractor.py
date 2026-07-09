"""
Extract HRV features using HeartPy ``hp.process()``.

Features: BPM, IBI, SDNN, SDSD, RMSSD, pNN20, pNN50, HR_MAD,
LF, HF, LF/HF, Breathing Rate, SD1, SD2, S.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from config.settings import HRV_FEATURE_COLUMNS, HRV_FEATURES, HRV_KEY_TO_COLUMN

logger = logging.getLogger(__name__)


def extract_hrv_features(
    segment: np.ndarray,
    sample_rate: float,
    *,
    bpmmin: int = 40,
    bpmmax: int = 180,
) -> dict[str, float] | None:
    """
    Run HeartPy on a single window and return the 15 HRV measures.

    Returns None if HeartPy raises or required measures are missing/NaN.
    """
    import heartpy as hp

    segment = np.asarray(segment, dtype=float).ravel()
    try:
        working_data, measures = hp.process(
            segment,
            sample_rate=sample_rate,
            bpmmin=bpmmin,
            bpmmax=bpmmax,
        )
    except Exception as exc:  # noqa: BLE001 — discard failed windows
        logger.debug("HeartPy failed on window: %s", exc)
        return None

    features: dict[str, float] = {}
    for key in HRV_FEATURES:
        col = HRV_KEY_TO_COLUMN[key]
        if key not in measures:
            logger.debug("Missing measure '%s'", key)
            return None
        val = measures[key]
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(fval):
            return None
        features[col] = fval

    return features


def extract_windows_hrv(
    windows: list[np.ndarray],
    sample_rate: float,
    *,
    record_id: str,
    subject_id: str,
    label: int,
    sensor: str,
    bpmmin: int = 40,
    bpmmax: int = 180,
) -> pd.DataFrame:
    """
    Extract HRV features for all windows; discard failures.

    Returns a DataFrame with HRV columns + metadata
    (record_id, subject_id, window_idx, label, sensor).
    """
    rows: list[dict[str, Any]] = []
    n_fail = 0

    for idx, win in enumerate(windows):
        feats = extract_hrv_features(
            win, sample_rate, bpmmin=bpmmin, bpmmax=bpmmax
        )
        if feats is None:
            n_fail += 1
            continue
        row = {
            "record_id": record_id,
            "subject_id": subject_id,
            "window_idx": idx,
            "label": int(label),
            "sensor": sensor,
            **feats,
        }
        rows.append(row)

    logger.info(
        "Record %s: %d/%d windows succeeded (%d discarded)",
        record_id,
        len(rows),
        len(windows),
        n_fail,
    )
    if not rows:
        return pd.DataFrame(columns=["record_id", "subject_id", "window_idx", "label", "sensor", *HRV_FEATURE_COLUMNS])

    return pd.DataFrame(rows)
