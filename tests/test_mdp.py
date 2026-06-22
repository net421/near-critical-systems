"""Tests for mdp.py: RVI, threshold evaluation, threshold search."""
from __future__ import annotations

import numpy as np

from buffer_policy.kernel import build_kernel
from buffer_policy.mdp import (
    evaluate_threshold_policy,
    is_threshold_policy,
    optimize_threshold,
    run_rvi,
)
from buffer_policy.params import CostParams, baseline_cost, baseline_model


def test_rvi_converges_baseline() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    res = run_rvi(mp, cp, tol=1e-11, max_iter=5000)
    assert res.converged, f"RVI did not converge; final span={res.final_span:.3e}"
    assert res.n_iter < 2000, f"RVI took {res.n_iter} iterations"
    assert np.isfinite(res.g) and res.g > 0.0


def test_rvi_policy_in_zero_one() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    res = run_rvi(mp, cp)
    assert set(np.unique(res.policy[1:])).issubset({0, 1})


def test_threshold_window_baseline() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    s_star, g_star, _ = optimize_threshold(mp, cp)
    assert 30 <= s_star <= 45, f"baseline s* = {s_star} outside [30, 45]"
    assert 0.0 < g_star < cp.K_p


def test_threshold_evaluate_specific_value() -> None:
    """g(s*=0) must equal g_RTF (never maintain)."""
    from buffer_policy.survival import g_run_to_failure_linear_system

    mp = baseline_model()
    cp = baseline_cost()
    P = build_kernel(mp)
    g_rtf = g_run_to_failure_linear_system(P, mp, cp.K_f)
    g_zero = evaluate_threshold_policy(0, mp, cp, P)
    rel = abs(g_zero - g_rtf) / g_rtf
    assert rel < 1e-9


def test_threshold_evaluate_reset_in_maintenance_region() -> None:
    """When s_star >= s_reset, g(s*) = K_p exactly."""
    mp = baseline_model()
    cp = baseline_cost()
    g = evaluate_threshold_policy(mp.s_reset, mp, cp)
    assert abs(g - cp.K_p) < 1e-12

    g_high = evaluate_threshold_policy(mp.S_max, mp, cp)
    assert abs(g_high - cp.K_p) < 1e-12


def test_threshold_evaluate_just_below_reset() -> None:
    """g(s*=s_reset-1) is finite, positive."""
    mp = baseline_model()
    cp = baseline_cost()
    g = evaluate_threshold_policy(mp.s_reset - 1, mp, cp)
    assert np.isfinite(g) and g > 0.0


def test_full_action_rvi_is_threshold() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    res = run_rvi(mp, cp)
    ok, s_star = is_threshold_policy(res.policy)
    assert ok, "RVI policy is not of threshold form"
    assert 0 <= s_star < mp.s_reset


def test_full_action_rvi_matches_threshold_search() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    res = run_rvi(mp, cp)
    s_star, g_star, _ = optimize_threshold(mp, cp)
    rel = abs(res.g - g_star) / g_star
    assert rel < 1e-6


def test_is_threshold_policy_helper() -> None:
    pol = np.array([0, 1, 1, 1, 1], dtype=np.int8)
    ok, s = is_threshold_policy(pol)
    assert ok and s == 4
    pol = np.array([0, 1, 1, 0, 0], dtype=np.int8)
    ok, s = is_threshold_policy(pol)
    assert ok and s == 2
    pol = np.array([0, 0, 0, 0, 0], dtype=np.int8)
    ok, s = is_threshold_policy(pol)
    assert ok and s == 0
    pol = np.array([0, 1, 0, 1, 0], dtype=np.int8)
    ok, s = is_threshold_policy(pol)
    assert not ok


def test_threshold_curve_finite_positive() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    _, _, gs = optimize_threshold(mp, cp)
    assert np.all(np.isfinite(gs))
    assert np.all(gs > 0.0)


def test_higher_kp_gives_lower_threshold() -> None:
    """Increasing K_p makes preventive maintenance less attractive."""
    mp = baseline_model()
    cp_low = CostParams(K_f=100.0, K_p=5.0)
    cp_high = CostParams(K_f=100.0, K_p=20.0)
    s_low, _, _ = optimize_threshold(mp, cp_low)
    s_high, _, _ = optimize_threshold(mp, cp_high)
    assert s_low >= s_high


def test_wide_search_caps_at_kp() -> None:
    """For s* in [s_reset, S_max], g(s*) = K_p exactly."""
    mp = baseline_model()
    cp = baseline_cost()
    P = build_kernel(mp)
    for s in (mp.s_reset, mp.s_reset + 10, mp.S_max):
        g = evaluate_threshold_policy(s, mp, cp, P)
        assert abs(g - cp.K_p) < 1e-12
