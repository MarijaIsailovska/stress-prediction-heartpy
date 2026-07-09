"""
Load EPHNOGRAM ECG and Wrist PPG recordings from PhysioNet formats.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from config.ephnogram_records import record_filename
from config.settings import EPHNOGRAM_RAW_DIR, WRIST_RAW_DIR
from config.wrist_records import WRIST_ACTIVITY_LABELS, activity_name_to_label

logger = logging.getLogger(__name__)


def _try_wfdb_rdsamp(record_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Read a WFDB record; return (signal, fields)."""
    import wfdb

    # wfdb expects path without extension
    stem = str(record_path.with_suffix(""))
    signals, fields = wfdb.rdsamp(stem)
    return np.asarray(signals, dtype=float), fields


def _load_mat_ecg(mat_path: Path) -> tuple[np.ndarray, float | None]:
    """
    Load ECG from MATLAB .mat file.

    EPHNOGRAM .mat files typically contain ECG (and PCG) under variable names
    such as 'ECG', 'ecg', or nested structs. We probe common keys.
    """
    from scipy.io import loadmat

    mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    keys = [k for k in mat.keys() if not k.startswith("__")]
    logger.debug("MAT keys in %s: %s", mat_path.name, keys)

    signal = None
    fs = None

    # Direct array keys
    for key in ("ECG", "ecg", "ECG_signal", "ecg_signal", "sig", "signal"):
        if key in mat:
            signal = np.asarray(mat[key], dtype=float).squeeze()
            break

    # Nested struct (common PhysioNet pattern)
    if signal is None:
        for key in keys:
            obj = mat[key]
            if hasattr(obj, "ECG"):
                signal = np.asarray(obj.ECG, dtype=float).squeeze()
            elif hasattr(obj, "ecg"):
                signal = np.asarray(obj.ecg, dtype=float).squeeze()
            if hasattr(obj, "fs"):
                fs = float(obj.fs)
            elif hasattr(obj, "Fs"):
                fs = float(obj.Fs)
            if signal is not None:
                break

    if signal is None:
        # Fallback: largest 1-D numeric array
        candidates = []
        for key in keys:
            arr = np.asarray(mat[key])
            if arr.ndim >= 1 and arr.size > 1000 and np.issubdtype(arr.dtype, np.number):
                candidates.append((arr.size, arr.squeeze()))
        if not candidates:
            raise ValueError(f"No ECG array found in {mat_path}")
        signal = max(candidates, key=lambda x: x[0])[1]

    signal = np.asarray(signal, dtype=float).ravel()
    return signal, fs


def load_ephnogram_ecg(
    record_id: str,
    data_dir: Path | None = None,
    channel: int = 0,
) -> tuple[np.ndarray, float]:
    """
    Load a single EPHNOGRAM ECG recording.

    Tries WFDB (.dat/.hea) first, then MATLAB (.mat).

    Parameters
    ----------
    record_id : str
        Two-digit record ID (e.g. '01', '10').
    data_dir : Path, optional
        Root directory containing ECGPCG00XY.* files.
    channel : int
        Channel index when multi-channel WFDB is loaded (ECG usually 0).

    Returns
    -------
    signal : np.ndarray
        1-D ECG samples at original sampling rate.
    fs : float
        Sampling frequency in Hz.
    """
    data_dir = Path(data_dir) if data_dir else EPHNOGRAM_RAW_DIR
    base = record_filename(record_id)

    hea = data_dir / f"{base}.hea"
    dat = data_dir / f"{base}.dat"
    mat = data_dir / f"{base}.mat"

    # Also search one level deeper (PhysioNet download layout)
    if not hea.exists() and not mat.exists():
        matches = list(data_dir.rglob(f"{base}.hea")) + list(data_dir.rglob(f"{base}.mat"))
        if matches:
            parent = matches[0].parent
            hea = parent / f"{base}.hea"
            dat = parent / f"{base}.dat"
            mat = parent / f"{base}.mat"

    if hea.exists() and dat.exists():
        signals, fields = _try_wfdb_rdsamp(hea)
        fs = float(fields["fs"])
        if signals.ndim == 2:
            # Prefer channel named ECG if present
            sig_names = [s.lower() for s in fields.get("sig_name", [])]
            if "ecg" in sig_names:
                channel = sig_names.index("ecg")
            signal = signals[:, channel]
        else:
            signal = signals.ravel()
        logger.info("Loaded WFDB %s (fs=%.1f, n=%d)", base, fs, len(signal))
        return signal.astype(float), fs

    if mat.exists():
        signal, fs_mat = _load_mat_ecg(mat)
        fs = float(fs_mat) if fs_mat else 8000.0
        logger.info("Loaded MAT %s (fs=%.1f, n=%d)", base, fs, len(signal))
        return signal, fs

    raise FileNotFoundError(
        f"EPHNOGRAM record {base} not found under {data_dir}. "
        "Download from https://physionet.org/content/ephnogram/1.0.0/ "
        "and place .mat or .dat/.hea files in data/raw/ephnogram/"
    )


def discover_wrist_records(data_dir: Path | None = None) -> list[dict[str, Any]]:
    """
    Discover Wrist PPG WFDB records under data/raw/wrist/.

    Returns a list of dicts with keys:
        path, subject_id, activity, activity_label
    """
    data_dir = Path(data_dir) if data_dir else WRIST_RAW_DIR
    records: list[dict[str, Any]] = []

    if not data_dir.exists():
        logger.warning("Wrist data directory missing: %s", data_dir)
        return records

    for hea in sorted(data_dir.rglob("*.hea")):
        stem = hea.stem.lower()
        parts = stem.replace("-", "_").split("_")
        subject_id = None
        activity = None

        for p in parts:
            if p in WRIST_ACTIVITY_LABELS:
                activity = p
            if p.startswith("s") and p[1:].isdigit():
                subject_id = p
            # patterns like subject1, subj1
            if p.startswith("subject") and p.replace("subject", "").isdigit():
                subject_id = f"s{p.replace('subject', '')}"

        # Infer from parent folder names
        for parent in hea.parents:
            name = parent.name.lower()
            if activity is None and name in WRIST_ACTIVITY_LABELS:
                activity = name
            if subject_id is None and name in {f"s{i}" for i in range(1, 9)}:
                subject_id = name
            if subject_id is None and name.startswith("s") and name[1:].isdigit():
                subject_id = name

        if activity is None:
            # Skip annotation-only or unknown files
            logger.debug("Skipping %s (no activity inferred)", hea)
            continue

        if subject_id is None:
            subject_id = hea.parent.name

        records.append(
            {
                "path": hea.with_suffix(""),
                "subject_id": subject_id,
                "activity": activity,
                "activity_label": activity_name_to_label(activity),
                "record_id": hea.stem,
            }
        )

    logger.info("Discovered %d wrist PPG records under %s", len(records), data_dir)
    return records


def load_wrist_ppg(
    record_path: Path | str,
    channel: int | None = None,
) -> tuple[np.ndarray, float]:
    """
    Load a Wrist PPG WFDB recording.

    Parameters
    ----------
    record_path : Path
        Path to record without extension, or path to .hea/.dat.
    channel : int, optional
        PPG channel index; auto-detected from signal names when possible.

    Returns
    -------
    signal : np.ndarray
        1-D PPG samples.
    fs : float
        Sampling frequency (expected 256 Hz).
    """
    record_path = Path(record_path)
    if record_path.suffix:
        record_path = record_path.with_suffix("")

    signals, fields = _try_wfdb_rdsamp(record_path)
    fs = float(fields["fs"])
    sig_names = [s.lower() for s in fields.get("sig_name", [])]

    if signals.ndim == 1:
        signal = signals.ravel()
    else:
        if channel is not None:
            idx = channel
        else:
            idx = 0
            for name in ("ppg", "pleth", "bvp", "wrist_ppg"):
                if name in sig_names:
                    idx = sig_names.index(name)
                    break
        signal = signals[:, idx]

    logger.info(
        "Loaded wrist PPG %s (fs=%.1f, n=%d, ch=%s)",
        record_path.name,
        fs,
        len(signal),
        sig_names[idx] if sig_names else idx,
    )
    return signal.astype(float), fs
