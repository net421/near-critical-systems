"""
Survival distribution of the collapse time tau under run-to-failure.

Given the absorbing kernel P (state 0 = collapse), the survival function is

    S(t) = sum_{s >= 1} mu_t(s),  mu_{t+1}(s') = sum_{s >= 1} mu_t(s) P[s, s']

with mu_0 concentrated at the initial state X0 (so S(0) = 1).

E[tau] = sum_{t=0}^{infty} S(t).

Two independent computations are provided:

* `survival_from_kernel` + `expected_tau`: forward propagation; returns the
  full vector S(0..T), needed for age-based costs and hazard diagnostics.
* `expected_tau_linear_system`:  closed-form solution v = (I - Q)^{-1} 1,
  used as a high-precision audit reference.

Note on `tol`: setting `tol < 0` disables early stopping and forces the
iteration to run for exactly T_max steps.  This is needed by the paper-grid
hazard pipeline, which requires a fixed-horizon survival vector.
"""
from __future__ import annotations

import numpy as np

from buffer_policy.params import ModelParams


def survival_from_kernel(
    P: np.ndarray,
    mp: ModelParams,
    T_max: int = 20_000,
    tol: float = 1e-15,
) -> np.ndarray:
    """
    Compute S(0), S(1), ..., S(T) where T is the smallest index with
    S(T) <= tol or T = T_max.  When tol < 0, early stopping is disabled
    and the result has length exactly T_max + 1.

    Returns
    -------
    S : np.ndarray
        S[0] = 1.  Monotone non-increasing.
    """
    if P.shape[0] != mp.S_max + 1:
        raise ValueError("Kernel shape inconsistent with ModelParams.S_max")

    Q = P[1:, 1:]
    mu = np.zeros(mp.S_max, dtype=np.float64)
    mu[mp.X0 - 1] = 1.0

    surv = [1.0]
    for _ in range(T_max):
        mu = mu @ Q
        s_t = float(mu.sum())
        if s_t < 0.0:
            s_t = 0.0
        surv.append(s_t)
        if tol >= 0.0 and s_t <= tol:
            break

    S = np.asarray(surv, dtype=np.float64)
    S = np.minimum.accumulate(S)
    return S


def expected_tau(S: np.ndarray) -> float:
    """E[tau] = sum_{t=0}^{infty} S(t).  Uses the truncated tail directly."""
    return float(np.sum(S))


def g_run_to_failure(S: np.ndarray, K_f: float) -> float:
    """g_RTF = K_f / E[tau], computed directly from the survival vector."""
    Etau = expected_tau(S)
    if Etau <= 0.0:
        raise ValueError(f"E[tau] = {Etau} is non-positive")
    return K_f / Etau


def expected_tau_linear_system(P: np.ndarray, mp: ModelParams) -> float:
    """
    Compute E[tau] exactly by solving (I - Q) v = 1, E[tau] = v(X0).

    Q is the sub-stochastic kernel on transient states {1, ..., S_max}.
    """
    if P.shape[0] != mp.S_max + 1:
        raise ValueError("Kernel shape inconsistent with ModelParams.S_max")

    Q = P[1:, 1:]
    n = mp.S_max
    A = np.eye(n) - Q
    b = np.ones(n)
    try:
        v = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as e:
        raise RuntimeError(
            f"Linear system (I-Q) v = 1 is singular: {e}.  "
            "This typically means the chain is not absorbing within S_max."
        ) from e
    return float(v[mp.X0 - 1])


def g_run_to_failure_linear_system(
    P: np.ndarray, mp: ModelParams, K_f: float
) -> float:
    """g_RTF computed via the exact linear-system E[tau].  Audit twin of g_RTF."""
    Etau = expected_tau_linear_system(P, mp)
    if Etau <= 0.0:
        raise ValueError(f"E[tau] = {Etau} is non-positive")
    return K_f / Etau
