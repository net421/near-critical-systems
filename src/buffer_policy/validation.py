"""
Structural audits for the MDP solution.

Lemma 1 (conditional)
---------------------
If g_RTF < K_p, then the optimal threshold s* satisfies s* < s_reset.

Wide-range audit: search s* over {0, ..., S_max}.  When s* >= s_reset,
g(s*) = K_p; the audit checks that the argmin still falls strictly below
s_reset.

Threshold-vs-RVI agreement
--------------------------
Full-action RVI must yield a threshold policy whose long-run cost agrees
with the best threshold from `optimize_threshold` to relative tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from buffer_policy.kernel import build_kernel
from buffer_policy.mdp import (
    RVIResult,
    extract_threshold_from_rvi,
    is_threshold_policy,
    optimize_threshold,
    run_rvi,
)
from buffer_policy.params import CostParams, ModelParams
from buffer_policy.survival import g_run_to_failure_linear_system


@dataclass(frozen=True)
class Lemma1Audit:
    g_RTF: float
    K_p: float
    s_star_wide: int
    g_star_wide: float
    s_reset: int
    premise_holds: bool
    holds: bool


def audit_lemma1(
    mp: ModelParams, cp: CostParams, P: np.ndarray | None = None
) -> Lemma1Audit:
    """Wide-range threshold search; verify s* < s_reset whenever g_RTF < K_p."""
    if P is None:
        P = build_kernel(mp)
    g_rtf = g_run_to_failure_linear_system(P, mp, cp.K_f)
    s_star, g_star, _ = optimize_threshold(
        mp, cp, P=P, s_min=0, s_max=mp.S_max
    )
    premise = g_rtf < cp.K_p
    if premise:
        holds = s_star < mp.s_reset
    else:
        holds = True
    return Lemma1Audit(
        g_RTF=g_rtf,
        K_p=cp.K_p,
        s_star_wide=s_star,
        g_star_wide=g_star,
        s_reset=mp.s_reset,
        premise_holds=premise,
        holds=holds,
    )


@dataclass(frozen=True)
class ThresholdRVIAudit:
    rvi_is_threshold: bool
    rvi_s_star: int | None
    g_rvi: float
    s_star_search: int
    g_star_search: float
    cost_rel_diff: float


def audit_threshold_vs_rvi(
    mp: ModelParams,
    cp: CostParams,
    P: np.ndarray | None = None,
    rvi: RVIResult | None = None,
) -> ThresholdRVIAudit:
    """Verify RVI returns a threshold policy with cost matching threshold search."""
    if P is None:
        P = build_kernel(mp)
    if rvi is None:
        rvi = run_rvi(mp, cp, P=P)
    rvi_s = extract_threshold_from_rvi(rvi)
    is_thresh, _ = is_threshold_policy(rvi.policy)

    s_star, g_star, _ = optimize_threshold(mp, cp, P=P)
    rel = abs(rvi.g - g_star) / max(abs(g_star), 1e-300)

    return ThresholdRVIAudit(
        rvi_is_threshold=is_thresh,
        rvi_s_star=rvi_s,
        g_rvi=rvi.g,
        s_star_search=s_star,
        g_star_search=g_star,
        cost_rel_diff=rel,
    )
