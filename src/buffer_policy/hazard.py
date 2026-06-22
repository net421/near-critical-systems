"""
Discrete hazard and DFR (decreasing failure rate) diagnostics.

    hazard[k] = (S[k] - S[k+1]) / S[k]    (for k where S[k] > survival_tol)

This modular version uses monotone_tol=1e-12 and approx_threshold=0.95.
The paper-grade DFR diagnostic lives in `experiments_paper.py` (Part 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def discrete_hazard(
    S: np.ndarray,
    survival_tol: float = 1e-8,
) -> np.ndarray:
    """Return the raw discrete hazard restricted to S(t) > survival_tol."""
    S = np.asarray(S, dtype=np.float64)
    if S.size < 2:
        return np.empty(0, dtype=np.float64)
    denom = S[:-1]
    h = (S[:-1] - S[1:]) / np.where(denom > 0.0, denom, 1.0)
    mask = denom > survival_tol
    return h[mask]


def moving_average(x: np.ndarray, window: int = 25) -> np.ndarray:
    """Moving average via convolution; returns length max(0, n-w+1)."""
    if window < 1:
        raise ValueError("window must be >= 1")
    x = np.asarray(x, dtype=np.float64)
    if x.size < window:
        return x.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(x, kernel, mode="valid")


@dataclass(frozen=True)
class HazardDiagnostics:
    DFR_strict_raw: bool
    DFR_strict_smoothed: bool
    DFR_approx_smoothed: bool
    frac_decreasing_raw: float
    frac_decreasing_smoothed: float
    h_initial: float
    h_final: float
    n_hazard_points: int
    n_hazard_smoothed: int
    smoothing_window: int


def hazard_diagnostics(
    S: np.ndarray,
    window: int = 25,
    survival_tol: float = 1e-8,
    monotone_tol: float = 1e-12,
    approx_threshold: float = 0.95,
) -> HazardDiagnostics:
    """Compute the full DFR diagnostic bundle for a survival vector S."""
    h_raw = discrete_hazard(S, survival_tol=survival_tol)
    n = h_raw.size
    if n < 2:
        return HazardDiagnostics(
            DFR_strict_raw=False,
            DFR_strict_smoothed=False,
            DFR_approx_smoothed=False,
            frac_decreasing_raw=float("nan"),
            frac_decreasing_smoothed=float("nan"),
            h_initial=float(h_raw[0]) if n == 1 else float("nan"),
            h_final=float(h_raw[-1]) if n == 1 else float("nan"),
            n_hazard_points=int(n),
            n_hazard_smoothed=0,
            smoothing_window=int(window),
        )

    diff_raw = np.diff(h_raw)
    dfr_strict_raw = bool(np.all(diff_raw <= monotone_tol))
    frac_dec_raw = float((diff_raw <= monotone_tol).mean())

    h_smooth = moving_average(h_raw, window=window)
    if h_smooth.size < 2:
        dfr_strict_smooth = False
        dfr_approx_smooth = False
        frac_dec_smooth = float("nan")
    else:
        diff_smooth = np.diff(h_smooth)
        dfr_strict_smooth = bool(np.all(diff_smooth <= monotone_tol))
        frac_dec_smooth = float((diff_smooth <= monotone_tol).mean())
        dfr_approx_smooth = bool(frac_dec_smooth >= approx_threshold)

    return HazardDiagnostics(
        DFR_strict_raw=dfr_strict_raw,
        DFR_strict_smoothed=dfr_strict_smooth,
        DFR_approx_smoothed=dfr_approx_smooth,
        frac_decreasing_raw=frac_dec_raw,
        frac_decreasing_smoothed=frac_dec_smooth,
        h_initial=float(h_raw[0]),
        h_final=float(h_raw[-1]),
        n_hazard_points=int(n),
        n_hazard_smoothed=int(h_smooth.size),
        smoothing_window=int(window),
    )
