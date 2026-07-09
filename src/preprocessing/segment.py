"""Window segmentation for HRV analysis."""

from __future__ import annotations

import numpy as np


def segment_signal(
    signal: np.ndarray,
    sample_rate: float,
    window_seconds: float = 60.0,
    overlap: float = 0.0,
) -> list[np.ndarray]:
    """
    Split a 1-D signal into fixed-length windows.

    Parameters
    ----------
    signal : np.ndarray
        Input samples.
    sample_rate : float
        Sampling rate (Hz).
    window_seconds : float
        Window length in seconds (default 60 for LF/HF).
    overlap : float
        Fractional overlap in [0, 1). 0.0 = non-overlapping.

    Returns
    -------
    windows : list of np.ndarray
        Contiguous segments of length ``int(window_seconds * sample_rate)``.
        Trailing incomplete windows are discarded.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    win_len = int(round(window_seconds * sample_rate))
    if win_len <= 0:
        raise ValueError("window_seconds * sample_rate must be > 0")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    step = max(1, int(round(win_len * (1.0 - overlap))))
    windows: list[np.ndarray] = []
    start = 0
    n = len(signal)
    while start + win_len <= n:
        windows.append(signal[start : start + win_len].copy())
        start += step
    return windows


def n_windows(
    n_samples: int,
    sample_rate: float,
    window_seconds: float = 60.0,
    overlap: float = 0.0,
) -> int:
    """Return how many full windows fit in ``n_samples``."""
    win_len = int(round(window_seconds * sample_rate))
    if win_len <= 0 or n_samples < win_len:
        return 0
    step = max(1, int(round(win_len * (1.0 - overlap))))
    return 1 + (n_samples - win_len) // step
