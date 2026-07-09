#!/usr/bin/env python3
"""
02 — Process Wrist PPG During Exercise into HRV feature CSV.

Pipeline per recording:
  load PPG @ 256 Hz → bandpass 0.75–3.5 Hz → 60 s windows
  → HeartPy hp.process() → discard failures
  → binary label: rest=0, walk/run/bike=1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import (
    PPG_BANDPASS,
    WINDOW_OVERLAP,
    WINDOW_SECONDS,
    WRIST_FEATURES_CSV,
    WRIST_FS,
    WRIST_RAW_DIR,
)
from config.wrist_records import to_binary_label
from src.data.loaders import discover_wrist_records, load_wrist_ppg
from src.features.hrv_extractor import extract_windows_hrv
from src.preprocessing.filter import bandpass_filter
from src.preprocessing.segment import segment_signal
from src.utils.io import ensure_dirs, save_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("02_process_wrist")


def process_one(meta: dict) -> pd.DataFrame:
    signal, fs = load_wrist_ppg(meta["path"])
    if abs(fs - WRIST_FS) > 1.0:
        logger.warning(
            "Record %s fs=%.1f (expected %d)", meta["record_id"], fs, WRIST_FS
        )

    signal = bandpass_filter(signal, fs, PPG_BANDPASS)
    windows = segment_signal(
        signal, fs, window_seconds=WINDOW_SECONDS, overlap=WINDOW_OVERLAP
    )
    binary_label = to_binary_label(meta["activity_label"])

    logger.info(
        "Record %s | subject=%s activity=%s → binary=%d | %d windows",
        meta["record_id"],
        meta["subject_id"],
        meta["activity"],
        binary_label,
        len(windows),
    )

    return extract_windows_hrv(
        windows,
        fs,
        record_id=meta["record_id"],
        subject_id=str(meta["subject_id"]),
        label=binary_label,
        sensor="ppg",
    )


def main() -> None:
    ensure_dirs()
    records = discover_wrist_records(WRIST_RAW_DIR)
    if not records:
        logger.error(
            "No wrist WFDB records found under %s. "
            "Download from https://physionet.org/content/wrist/1.0.0/",
            WRIST_RAW_DIR,
        )
        sys.exit(1)

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for meta in records:
        try:
            df = process_one(meta)
            if not df.empty:
                frames.append(df)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed %s: %s", meta.get("record_id"), exc)
            errors.append(str(meta.get("record_id")))

    if not frames:
        logger.error("No features extracted from wrist PPG.")
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    save_dataframe(out, WRIST_FEATURES_CSV)
    logger.info(
        "Done: %d samples from %d records (%d errors) → %s",
        len(out),
        len(records) - len(errors),
        len(errors),
        WRIST_FEATURES_CSV,
    )


if __name__ == "__main__":
    main()
