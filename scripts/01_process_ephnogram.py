#!/usr/bin/env python3
"""
01 — Process EPHNOGRAM ECG recordings into HRV feature CSV.

Pipeline per recording (exact study specification):
  wfdb load (ch0) → resample_poly 8000→500 Hz → HeartPy bandpass 0.5–45 Hz
  → 60 s non-overlapping windows → hp.process(calc_freq=True)
  → QC filters → protocol label → CSV
"""

from __future__ import annotations

import sys
import warnings
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import heartpy as hp
import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample_poly

from config.ephnogram_records import (
    ALL_RECORDS,
    RECORD_LABELS,
    get_subject,
    record_filename,
    refresh_subject_map,
)

# ---------------------------------------------------------------------------
# Paths & constants (study specification)
# ---------------------------------------------------------------------------
WFDB_DIR = ROOT / "data" / "raw" / "ephnogram" / "WFDB"
OUT_CSV = ROOT / "data" / "processed" / "ephnogram_features.csv"

FS_TARGET = 500.0
BANDPASS = [0.5, 45.0]
WINDOW_SECONDS = 60
ECG_CHANNEL = 0

MIN_BEATS = 20
BPM_MIN, BPM_MAX = 30.0, 220.0
RMSSD_MAX_MS = 400.0  # extreme outlier only (was 300; see research_notes QC revision)
RMSSD_IBI_RATIO_MAX = 3.0  # reject if RMSSD > 3 * IBI (false-peak / impossible)
REST_BPM_MAX = 120.0  # rest label only: reject physiologically implausible HR

REJECT_KEYS = (
    "heartpy_exception",
    "beats_too_few",
    "bpm_out_of_range",
    "rmssd_too_high",
    "rmssd_exceeds_ibi",
    "rest_bpm_too_high",
    "nonfinite_or_empty_rr",
)

FEATURE_COLS = [
    "recording_id",
    "subject_id",
    "window_id",
    "label",
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


def empty_reject_counts() -> dict[str, int]:
    return {k: 0 for k in REJECT_KEYS}


def load_ecg(record_id: str) -> tuple[np.ndarray, float]:
    """Step 1 — Load ECG channel 0 via wfdb.rdrecord from WFDB/."""
    stem = WFDB_DIR / record_filename(record_id)
    if not stem.with_suffix(".dat").exists() or not stem.with_suffix(".hea").exists():
        raise FileNotFoundError(f"Missing WFDB files for {stem.name} under {WFDB_DIR}")

    rec = wfdb.rdrecord(str(stem))
    fs = float(rec.fs)
    sig = np.asarray(rec.p_signal, dtype=float)
    if sig.ndim == 1:
        ecg = sig
    else:
        ecg = sig[:, ECG_CHANNEL]
    return ecg.ravel(), fs


def downsample_to_500(signal: np.ndarray, fs: float) -> tuple[np.ndarray, float]:
    """Step 2 — Polyphase downsample to 500 Hz."""
    if abs(fs - FS_TARGET) < 1e-6:
        return signal.astype(float), FS_TARGET
    up = int(round(FS_TARGET))
    down = int(round(fs))
    g = int(np.gcd(up, down))
    up //= g
    down //= g
    out = resample_poly(signal, up, down)
    return np.asarray(out, dtype=float), FS_TARGET


def bandpass_ecg(signal: np.ndarray, sample_rate: float) -> np.ndarray:
    """Step 3 — HeartPy bandpass 0.5–45 Hz."""
    return hp.filter_signal(
        signal,
        cutoff=BANDPASS,
        sample_rate=sample_rate,
        filtertype="bandpass",
    )


def segment_60s(signal: np.ndarray, sample_rate: float) -> list[np.ndarray]:
    """Step 4 — Non-overlapping 60 s windows; drop trailing incomplete window."""
    win_len = int(round(WINDOW_SECONDS * sample_rate))
    n = len(signal)
    return [signal[i : i + win_len].copy() for i in range(0, n - win_len + 1, win_len)]


def _finite(x) -> float | None:
    try:
        if x is None:
            return None
        if np.ma.is_masked(x):
            return None
        v = float(np.asarray(x).item() if hasattr(np.asarray(x), "item") else x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def extract_window_features(
    segment: np.ndarray,
    sample_rate: float,
    label: int,
) -> tuple[dict | None, str | None]:
    """
    Steps 5–6 — HeartPy HRV + QC.

    Returns (features, None) on success, or (None, reject_reason_key).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wd, m = hp.process(
                segment,
                sample_rate=sample_rate,
                calc_freq=True,
            )
    except Exception:
        return None, "heartpy_exception"

    rr = np.asarray(wd.get("RR_list", []), dtype=float)
    peaks = wd.get("peaklist", [])
    n_beats = (
        len(peaks)
        if peaks is not None and len(peaks) > 0
        else (len(rr) + 1 if len(rr) else 0)
    )

    if n_beats < MIN_BEATS:
        return None, "beats_too_few"

    bpm = _finite(m.get("bpm"))
    ibi = _finite(m.get("ibi"))
    sdnn = _finite(m.get("sdnn"))
    rmssd = _finite(m.get("rmssd"))
    sdsd = _finite(m.get("sdsd"))
    pnn20 = _finite(m.get("pnn20"))
    pnn50 = _finite(m.get("pnn50"))
    sd1 = _finite(m.get("sd1"))
    sd2 = _finite(m.get("sd2"))
    breathing = _finite(m.get("breathingrate"))

    required = [bpm, ibi, sdnn, rmssd, sdsd, pnn20, pnn50, sd1, sd2, breathing]
    if any(v is None for v in required):
        return None, "nonfinite_or_empty_rr"

    if bpm < BPM_MIN or bpm > BPM_MAX:
        return None, "bpm_out_of_range"
    if rmssd > RMSSD_MAX_MS:
        return None, "rmssd_too_high"
    # Mathematically implausible: RMSSD cannot exceed ~mean RR (IBI) by a large factor
    if ibi > 0 and rmssd > RMSSD_IBI_RATIO_MAX * ibi:
        return None, "rmssd_exceeds_ibi"

    # Rest-only: BPM > 120 is physiologically implausible for protocol rest
    if label == 0 and bpm > REST_BPM_MAX:
        return None, "rest_bpm_too_high"

    if len(rr) == 0:
        return None, "nonfinite_or_empty_rr"
    hr_mad = float(np.median(np.abs(rr - np.median(rr))))
    if not np.isfinite(hr_mad):
        return None, "nonfinite_or_empty_rr"

    return (
        {
            "bpm": bpm,
            "ibi": ibi,
            "sdnn": sdnn,
            "rmssd": rmssd,
            "sdsd": sdsd,
            "pnn20": pnn20,
            "pnn50": pnn50,
            "hr_mad": hr_mad,
            "sd1": sd1,
            "sd2": sd2,
            "breathing_rate": breathing,
        },
        None,
    )


def process_recording(record_id: str) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Full pipeline for one recording ID. Returns (df, reject_counts, n_windows)."""
    recording_id = record_filename(record_id)
    label = int(RECORD_LABELS[record_id])
    subject_id = get_subject(record_id)
    rejects = empty_reject_counts()

    raw, fs = load_ecg(record_id)
    signal, fs_out = downsample_to_500(raw, fs)
    signal = bandpass_ecg(signal, fs_out)
    windows = segment_60s(signal, fs_out)

    rows: list[dict] = []
    for window_id, win in enumerate(windows):
        feats, reason = extract_window_features(win, fs_out, label)
        if reason is not None:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        rows.append(
            {
                "recording_id": recording_id,
                "subject_id": subject_id,
                "window_id": window_id,
                "label": label,
                **feats,
            }
        )

    n_rej = sum(rejects.values())
    print(
        f"{recording_id} | label={label} | subject={subject_id} | "
        f"windows_kept={len(rows)}/{len(windows)} | rejected={n_rej}"
    )
    print(f"  rejects: {rejects}")

    if not rows:
        return pd.DataFrame(columns=FEATURE_COLS), rejects, len(windows)
    return pd.DataFrame(rows, columns=FEATURE_COLS), rejects, len(windows)


def print_summary(df: pd.DataFrame, total_rejects: dict[str, int]) -> None:
    print("\n========== SUMMARY ==========")
    print(f"Total windows kept: {len(df)}")
    print("Class distribution (label):")
    print(df["label"].value_counts().rename({0: "rest(0)", 1: "stress(1)"}).to_string())

    print("\nTotal rejection reasons:")
    for k in REJECT_KEYS:
        print(f"  {k}: {total_rejects.get(k, 0)}")
    print(f"  TOTAL rejected: {sum(total_rejects.values())}")

    print("\nMissing / NaN per column:")
    print(df.isna().sum().to_string())

    feat_cols = [
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
    print("\nFeature mean ± std by class:")
    for lab, name in [(0, "rest"), (1, "stress")]:
        sub = df[df["label"] == lab]
        print(f"\n--- {name} (n={len(sub)}) ---")
        if sub.empty:
            continue
        stats = sub[feat_cols].agg(["mean", "std"]).T
        print(stats.to_string())


def main() -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not WFDB_DIR.is_dir():
        print(f"ERROR: WFDB directory not found: {WFDB_DIR}", file=sys.stderr)
        sys.exit(1)

    refresh_subject_map()
    print(f"Processing {len(ALL_RECORDS)} recordings from {WFDB_DIR}")
    print(f"Output -> {OUT_CSV}\n")

    frames: list[pd.DataFrame] = []
    failed: list[str] = []
    total_rejects: Counter[str] = Counter()

    for rid in ALL_RECORDS:
        try:
            df, rejects, _n_win = process_recording(rid)
            total_rejects.update(rejects)
            if not df.empty:
                frames.append(df)
            else:
                print(f"  WARNING: no valid windows for {record_filename(rid)}")
        except FileNotFoundError as exc:
            print(f"  ERROR: {exc}")
            failed.append(rid)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR processing {rid}: {exc}")
            failed.append(rid)

    if not frames:
        print("ERROR: no features extracted.", file=sys.stderr)
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    out = out[FEATURE_COLS]
    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}")
    print_summary(out, dict(total_rejects))

    if failed:
        print(f"\nFailed recordings ({len(failed)}): {failed}")


if __name__ == "__main__":
    main()
