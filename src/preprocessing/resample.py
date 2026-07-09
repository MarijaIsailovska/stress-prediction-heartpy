"""Resampling utilities."""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


def downsample(
    signal: np.ndarray,
    fs_original: float,
    fs_target: float,
) -> tuple[np.ndarray, float]:
    """
    Downsample a 1-D signal using polyphase resampling.

    Parameters
    ----------
    signal : np.ndarray
        Input samples.
    fs_original : float
        Original sampling rate (Hz).
    fs_target : float
        Desired sampling rate (Hz).

    Returns
    -------
    resampled : np.ndarray
    fs_out : float
        Equals fs_target (or fs_original if already at/below target).
    """
    signal = np.asarray(signal, dtype=float).ravel()
    if fs_target >= fs_original:
        return signal, float(fs_original)

    # Integer up/down factors via GCD for exact ratio when possible
    up = int(round(fs_target))
    down = int(round(fs_original))
    g = np.gcd(up, down)
    up //= g
    down //= g

    resampled = resample_poly(signal, up, down)
    return np.asarray(resampled, dtype=float), float(fs_target)
