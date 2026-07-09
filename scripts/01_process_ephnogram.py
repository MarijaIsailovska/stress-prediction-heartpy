#!/usr/bin/env python3
"""
01 — Process EPHNOGRAM ECG recordings into HRV feature CSV.

Pipeline per recording:
  load → downsample 8000→500 Hz → bandpass 0.5–45 Hz → 60 s windows
  → HeartPy hp.process() → discard failures → assign protocol label
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Project root on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config.ephnogram_records import (
    ALL_RECORDS,
    RECORD_LABELS,
    get_subject,
    refresh_subject_map,
)
from config.settings import (
    ECG_BANDPASS,
    EPHNOGRAM_FEATURES_CSV,
    EPHNOGRAM_FS_ORIGINAL,
    EPHNOGRAM_FS_TARGET,
    EPHNOGRAM_RAW_DIR,
    WINDOW_OVERLAP,
    WINDOW_SECONDS,
)
from src.data.loaders import load_ephnogram_ecg
from src.features.hrv_extractor import extract_windows_hrv
from src.preprocessing.filter import bandpass_filter
from src.preprocessing.resample import downsample
from src.preprocessing.segment import segment_signal
from src.utils.io import ensure_dirs, save_dataframe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("01_process_ephnogram")


def process_record(record_id: str) -> pd.DataFrame:
    label = RECORD_LABELS[record_id]
    subject_id = get_subject(record_id)

    signal, fs = load_ephnogram_ecg(record_id, data_dir=EPHNOGRAM_RAW_DIR)
    if abs(fs - EPHNOGRAM_FS_ORIGINAL) > 1.0:
        logger.warning(
            "Record %s fs=%.1f (expected %d); still resampling to %d Hz",
            record_id,
            fs,
            EPHNOGRAM_FS_ORIGINAL,
            EPHNOGRAM_FS_TARGET,
        )

    signal, fs_out = downsample(signal, fs, EPHNOGRAM_FS_TARGET)
    signal = bandpass_filter(signal, fs_out, ECG_BANDPASS)
    windows = segment_signal(
        signal, fs_out, window_seconds=WINDOW_SECONDS, overlap=WINDOW_OVERLAP
    )
    logger.info(
        "Record %s | subject=%s label=%d | %d windows @ %d Hz",
        record_id,
        subject_id,
        label,
        len(windows),
        int(fs_out),
    )

    return extract_windows_hrv(
        windows,
        fs_out,
        record_id=record_id,
        subject_id=subject_id,
        label=label,
        sensor="ecg",
    )


def main() -> None:
    ensure_dirs()
    if not EPHNOGRAM_RAW_DIR.exists():
        logger.error(
            "Missing %s — download EPHNOGRAM from PhysioNet and place files there.",
            EPHNOGRAM_RAW_DIR,
        )
        sys.exit(1)

    subject_map = refresh_subject_map()
    logger.info("Subject map entries: %d", len(subject_map))

    frames: list[pd.DataFrame] = []
    errors: list[str] = []

    for rid in ALL_RECORDS:
        try:
            df = process_record(rid)
            if not df.empty:
                frames.append(df)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            errors.append(rid)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed record %s: %s", rid, exc)
            errors.append(rid)

    if not frames:
        logger.error("No features extracted. Check raw data and HeartPy install.")
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    save_dataframe(out, EPHNOGRAM_FEATURES_CSV)
    logger.info(
        "Done: %d samples from %d/%d records (%d errors) → %s",
        len(out),
        len(ALL_RECORDS) - len(errors),
        len(ALL_RECORDS),
        len(errors),
        EPHNOGRAM_FEATURES_CSV,
    )
    if errors:
        logger.warning("Failed records: %s", errors)


if __name__ == "__main__":
    main()
