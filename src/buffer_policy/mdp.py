"""
Average-cost MDP solver for the buffer-threshold problem.

State space: {0, 1, ..., S_max} with state 0 as collapse marker.
Actions: a=0 (continue), a=1 (preventive maintenance -> jump to s_reset).

Bellman recursion (average-cost RVI):

    V_cont(s) = K_f * q_fail(s)
              + sum_{s' >= 1} P[s, s'] h(s')
              + P[s, 0] h(s_reset)               # reroute collapse mass
    V_prev(s) = K_p + h(s_reset)
    h_new(s)  = min(V_cont(s), V_prev(s))
    g         = h_new(s_reset)                   # reference: h(s_reset) = 0
    h_new    -= g

Convergence: span(h_new - h) < 1e-11.

Threshold policy evaluation
---------------------------
Under threshold s*:
    pi(s) = 1 (maintain) if s <= s*
    pi(s) = 0 (continue) if s >  s*

Regenerative cycle starts at s_reset and ends at the FIRST of:
    (a) collapse during a "continue" step                 -> pay K_f
    (b) entering the maintain set {s : s <= s*}, after which the next
        period executes preventive maintenance and returns to s_reset
                                                          -> pay K_p

Let C = {s : s > s*} be the continue set, Q the (sub-stochastic) restriction
of P to C x C, and r_f(s) = P[s, 0] for s in C.  Define F = (I - Q)^{-1}.
Starting from s_reset (assumed in C):

    E[N]            = (F 1)(s_reset)
    p_f             = (F r_f)(s_reset)
    p_p             = 1 - p_f
    E[cycle length] = E[N] + p_p
    E[cycle cost]   = K_f * p_f + K_p * p_p
    g(s*)           = E[cycle cost] / E[cycle length]

Edge case: if s_reset <= s*, then at reset the policy maintains every period,
so g(s*) = K_p.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from buffer_policy.kernel import build_kernel, q_fail_vector
from buffer_policy.params import CostParams, ModelParams


# --------------------------------------------------------------- RVI
@dataclass(frozen=True)
class RVIResult:
    """Output of relative value iteration."""
    g: float
    h: np.ndarray
    policy: np.ndarray
    n_iter: int
    converged: bool
    final_span: float


def run_rvi(
    mp: ModelParams,
    cp: CostParams,
    P: np.ndarray | None = None,
    tol: float = 1e-11,
    max_iter: int = 5000,
) -> RVIResult:
    """Solve the average-cost MDP via relative value iteration."""
    if P is None:
        P = build_kernel(mp)
    S = mp.S_max
    s_reset = mp.s_reset
    K_f, K_p = cp.K_f, cp.K_p

    qf = q_fail_vector(mp)
    h = np.zeros(S + 1, dtype=np.float64)
    policy = np.zeros(S + 1, dtype=np.int8)

    P_no0 = P[:, 1:]
    P_to0 = P[:, 0]

    converged = False
    final_span = np.inf
    n_iter = 0
    g_final = 0.0

    for it in range(1, max_iter + 1):
        cont = (
            K_f * qf[1:]
            + P_no0[1:] @ h[1:]
            + P_to0[1:] * h[s_reset]
        )
        prev_val = K_p + h[s_reset]

        h_new_active = np.minimum(cont, prev_val)
        new_policy_active = (prev_val < cont).astype(np.int8)

        h_new = np.zeros_like(h)
        h_new[1:] = h_new_active

        g_offset = h_new[s_reset]
        h_new -= g_offset
        h_new[0] = 0.0

        diff = h_new[1:] - h[1:]
        span_diff = float(diff.max() - diff.min())
        h = h_new
        policy[1:] = new_policy_active

        n_iter = it
        g_final = float(g_offset)
        if span_diff < tol:
            converged = True
            final_span = span_diff
            break
        final_span = span_diff

    return RVIResult(
        g=g_final,
        h=h,
        policy=policy,
        n_iter=n_iter,
        converged=converged,
        final_span=final_span,
    )


# --------------------------------------------------------------- Threshold evaluation
def evaluate_threshold_policy(
    s_star: int,
    mp: ModelParams,
    cp: CostParams,
    P: np.ndarray | None = None,
) -> float:
    """Long-run average cost g(s*) of the threshold policy."""
    if P is None:
        P = build_kernel(mp)
    S = mp.S_max
    if not (0 <= s_star <= S):
        raise ValueError(f"s_star must be in [0, S_max]; got {s_star}")
    if not (1 <= mp.s_reset <= S):
        raise ValueError("s_reset out of range")

    if mp.s_reset <= s_star:
        return float(cp.K_p)

    Q_full = P[1:, 1:]
    P_to0 = P[1:, 0]

    cont_mask = np.zeros(S, dtype=bool)
    cont_mask[s_star:] = True

    Q_pi = np.zeros((S, S), dtype=np.float64)
    Q_pi[cont_mask, :] = Q_full[cont_mask, :]
    if s_star > 0:
        Q_pi[cont_mask, :s_star] = 0.0

    A = np.eye(S) - Q_pi
    one = np.ones(S)
    v_steps = np.linalg.solve(A, one)
    s_idx = mp.s_reset - 1
    E_N = float(v_steps[s_idx])

    r_f = np.zeros(S)
    r_f[cont_mask] = P_to0[cont_mask]
    w = np.linalg.solve(A, r_f)
    p_f = float(w[s_idx])
    p_f = max(0.0, min(1.0, p_f))
    p_p = 1.0 - p_f

    cycle_len = E_N + p_p
    if cycle_len <= 0.0:
        return float("inf")
    cycle_cost = cp.K_f * p_f + cp.K_p * p_p
    return cycle_cost / cycle_len


def optimize_threshold(
    mp: ModelParams,
    cp: CostParams,
    P: np.ndarray | None = None,
    s_min: int = 0,
    s_max: int | None = None,
) -> tuple[int, float, np.ndarray]:
    """Search s* minimising g(s*) over s* in [s_min, s_max]."""
    if P is None:
        P = build_kernel(mp)
    if s_max is None:
        s_max = mp.s_reset - 1
    grid = np.arange(s_min, s_max + 1, dtype=int)
    gs = np.array([evaluate_threshold_policy(int(s), mp, cp, P) for s in grid])
    i = int(np.argmin(gs))
    return int(grid[i]), float(gs[i]), gs


# --------------------------------------------------------------- helpers
def is_threshold_policy(policy: np.ndarray) -> tuple[bool, int | None]:
    """Check whether policy is of threshold form."""
    pol = policy[1:]
    S = len(pol)
    if not pol.any():
        return True, 0
    s_star = int(np.where(pol == 1)[0].max() + 1)
    expected = np.zeros(S, dtype=np.int8)
    expected[:s_star] = 1
    return bool(np.array_equal(pol, expected)), s_star


def extract_threshold_from_rvi(rvi: RVIResult) -> int | None:
    """Return s* if RVI policy is threshold, else None."""
    ok, s_star = is_threshold_policy(rvi.policy)
    return s_star if ok else None
