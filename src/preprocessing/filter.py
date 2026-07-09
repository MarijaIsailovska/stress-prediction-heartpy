"""Bandpass filtering via HeartPy helpers (with SciPy fallback)."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def bandpass_filter(
    signal: np.ndarray,
    sample_rate: float,
    cutoff: tuple[float, float],
    order: int = 3,
) -> np.ndarray:
    """
    Apply a bandpass filter suitable for ECG or PPG.

    Prefers ``heartpy.filter_signal``; falls back to a Butterworth
    filtfilt implementation if HeartPy is unavailable.

    Parameters
    ----------
    signal : np.ndarray
        1-D input.
    sample_rate : float
        Sampling rate in Hz.
    cutoff : (low, high)
        Passband edges in Hz. ECG: (0.5, 45); PPG: (0.75, 3.5).
    order : int
        Filter order.
    """
    signal = np.asarray(signal, dtype=float).ravel()
    low, high = cutoff

    try:
        import heartpy as hp

        return hp.filter_signal(
            signal,
            cutoff=[low, high],
            sample_rate=sample_rate,
            order=order,
            filtertype="bandpass",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HeartPy filter_signal failed (%s); using SciPy Butterworth", exc)
        return _butter_bandpass(signal, sample_rate, low, high, order=order)


def _butter_bandpass(
    signal: np.ndarray,
    sample_rate: float,
    low: float,
    high: float,
    order: int = 3,
) -> np.ndarray:
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * sample_rate
    low_n = max(low / nyq, 1e-5)
    high_n = min(high / nyq, 0.999)
    if low_n >= high_n:
        raise ValueError(f"Invalid bandpass relative to Nyquist: {low}-{high} Hz @ {sample_rate} Hz")
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)
