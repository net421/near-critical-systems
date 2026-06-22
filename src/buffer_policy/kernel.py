"""
Capped MDP kernel construction.

State space: {0, 1, ..., S_max}.  State 0 is a COLLAPSE MARKER (absorbing in
this representation).  For s in {1, ..., S_max} and action a=0 (continue):

    P[s, 0]   = q_fail(s) = (1-p) P(D >= s+C) + p P(D >= s)

    Disruptive mode (Y=0):
        for s' in {1, ..., s}: P[s, s'] += p * P(D = s - s')

    Operational mode (Y=C):
        for s' in {1, ..., S_max - 1}: P[s, s'] += (1-p) * P(D = s+C - s')

    Upper boundary (s' = S_max):
        if s + C - S_max >= 0:
            P[s, S_max] += (1-p) * P(D <= s + C - S_max)

Row sums must equal 1 (audited, NOT silently normalised).
"""
from __future__ import annotations

import numpy as np

from buffer_policy.params import ModelParams
from buffer_policy.poisson_utils import (
    poisson_cdf,
    poisson_tail_ge,
    truncated_pmf_array,
)


def build_kernel(mp: ModelParams, audit_tol: float = 1e-12) -> np.ndarray:
    """
    Build the (S_max+1) x (S_max+1) transition matrix P.

    Parameters
    ----------
    mp : ModelParams
    audit_tol : float
        Maximum allowed |row_sum - 1| before raising.

    Returns
    -------
    P : np.ndarray, shape (S_max+1, S_max+1)
        Row-stochastic.  P[0, 0] = 1 (collapse absorbing in this kernel).
    """
    S = mp.S_max
    p, C, lam = mp.p, mp.C, mp.lam
    P = np.zeros((S + 1, S + 1), dtype=np.float64)

    # State 0 is absorbing in this raw kernel (collapse marker).  RVI/survival
    # logic re-routes mass entering state 0 elsewhere as needed.
    P[0, 0] = 1.0

    # Pre-compute pmf array up to the largest argument we will ever request.
    n_max = S + C
    pmf = truncated_pmf_array(lam, n_max)

    for s in range(1, S + 1):
        # ---- collapse probability ------------------------------------
        tail_op = poisson_tail_ge(s + C, lam)
        tail_dis = poisson_tail_ge(s, lam)
        q_fail = (1.0 - p) * tail_op + p * tail_dis
        P[s, 0] = q_fail

        # ---- disruptive mode (Y=0): s' in {1, ..., s} ----------------
        sprimes = np.arange(1, s + 1)
        ks = s - sprimes
        P[s, sprimes] += p * pmf[ks]

        # ---- operational mode (Y=C): s' in {1, ..., S_max - 1} -------
        if S - 1 >= 1:
            sprimes = np.arange(1, S)
            ks = s + C - sprimes
            mask = (ks >= 0) & (ks <= n_max)
            P[s, sprimes[mask]] += (1.0 - p) * pmf[ks[mask]]

        # ---- upper boundary at s' = S_max ----------------------------
        m = s + C - S
        if m >= 0:
            P[s, S] += (1.0 - p) * poisson_cdf(m, lam)

    # ----- audit row sums -------------------------------------------------
    row_sums = P.sum(axis=1)
    err = float(np.max(np.abs(row_sums - 1.0)))
    if err > audit_tol:
        bad = int(np.argmax(np.abs(row_sums - 1.0)))
        raise ValueError(
            f"Kernel row-sum audit failed: max |row_sum - 1| = {err:.3e} "
            f"at state {bad} (row sum = {row_sums[bad]:.16f}); "
            f"tolerance = {audit_tol:.1e}"
        )
    return P


def q_fail_vector(mp: ModelParams) -> np.ndarray:
    """
    Return q_fail[s] = P(I(t+1) <= 0 | I(t) = s) for s in {0,...,S_max}.

    q_fail[0] is set to 1 (state 0 is collapse).  For s >= 1:
        q_fail(s) = (1-p) P(D >= s+C) + p P(D >= s)
    """
    p, C, lam = mp.p, mp.C, mp.lam
    s = np.arange(0, mp.S_max + 1)
    out = np.empty_like(s, dtype=np.float64)
    out[0] = 1.0
    s1 = s[1:]
    out[1:] = (
        (1.0 - p) * poisson_tail_ge(s1 + C, lam)
        + p * poisson_tail_ge(s1, lam)
    )
    return out
