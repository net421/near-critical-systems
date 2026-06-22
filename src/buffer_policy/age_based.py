"""
Age-based maintenance policy: replace at age T or upon failure.

Given survival S(t) = P(tau > t),

    g_AB(T) = [ K_f * (1 - S(T)) + K_p * S(T) ] / sum_{t=0}^{T-1} S(t).

As T -> infty, S(T) -> 0 and the denominator -> E[tau], so g_AB -> K_f / E[tau]
= g_RTF.
"""
from __future__ import annotations

import numpy as np


def age_based_cost(S: np.ndarray, T: int, K_f: float, K_p: float) -> float:
    """g_AB(T) for a single horizon T >= 1."""
    if T < 1:
        raise ValueError("T must be >= 1")
    if T >= len(S):
        S_T = 0.0
        denom = float(np.sum(S))
    else:
        S_T = float(S[T])
        denom = float(np.sum(S[:T]))
    if denom <= 0.0:
        raise ValueError("Denominator sum_{t<T} S(t) is non-positive")
    return (K_f * (1.0 - S_T) + K_p * S_T) / denom


def age_based_curve(
    S: np.ndarray, T_grid: np.ndarray, K_f: float, K_p: float
) -> np.ndarray:
    """
    Vectorised g_AB(T) for an array of horizons.

    Same formula as age_based_cost(), but uses cumulative sums so dense
    grids such as T=1..20_000 are much faster.
    """
    S = np.asarray(S, dtype=np.float64)
    T_grid = np.asarray(T_grid, dtype=int)

    if np.any(T_grid < 1):
        raise ValueError("All T must be >= 1")

    csum = np.concatenate(([0.0], np.cumsum(S)))
    total = float(csum[-1])
    out = np.empty(T_grid.shape, dtype=np.float64)

    for i, T in enumerate(T_grid):
        if T >= len(S):
            S_T = 0.0
            denom = total
        else:
            S_T = float(S[T])
            denom = float(csum[T])

        if denom <= 0.0:
            raise ValueError("Denominator sum_{t<T} S(t) is non-positive")

        out[i] = (K_f * (1.0 - S_T) + K_p * S_T) / denom

    return out


def optimal_age(
    S: np.ndarray, K_f: float, K_p: float, T_max: int | None = None
) -> tuple[int, float]:
    """Return (T*, g_AB(T*)) minimising g_AB over T in {1, ..., T_max}."""
    if T_max is None:
        T_max = len(S) - 1
    Ts = np.arange(1, T_max + 1)
    g = age_based_curve(S, Ts, K_f, K_p)
    i = int(np.argmin(g))
    return int(Ts[i]), float(g[i])
