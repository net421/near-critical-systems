"""Survival + age-based cost tests."""
from __future__ import annotations

import numpy as np

from buffer_policy.age_based import age_based_cost, age_based_curve
from buffer_policy.kernel import build_kernel
from buffer_policy.params import baseline_cost, baseline_model
from buffer_policy.survival import (
    expected_tau,
    expected_tau_linear_system,
    g_run_to_failure,
    g_run_to_failure_linear_system,
    survival_from_kernel,
)


def test_survival_starts_at_one_and_monotone() -> None:
    mp = baseline_model()
    P = build_kernel(mp)
    S = survival_from_kernel(P, mp, T_max=20_000)
    assert S[0] == 1.0
    diffs = np.diff(S)
    assert np.all(diffs <= 1e-15)


def test_expected_tau_positive_and_finite() -> None:
    mp = baseline_model()
    P = build_kernel(mp)
    S = survival_from_kernel(P, mp, T_max=50_000)
    Etau = expected_tau(S)
    assert 1.0 < Etau < 1e10


def test_g_rtf_sensible() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    P = build_kernel(mp)
    S = survival_from_kernel(P, mp, T_max=50_000)
    g = g_run_to_failure(S, cp.K_f)
    assert g > 0.0
    assert g < cp.K_f


def test_age_based_limit_recovers_g_rtf() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    P = build_kernel(mp)
    S = survival_from_kernel(P, mp, T_max=200_000, tol=1e-18)
    g_rtf = g_run_to_failure(S, cp.K_f)
    T = min(len(S) - 1, 10_000)
    g_ab = age_based_cost(S, T, cp.K_f, cp.K_p)
    rel = abs(g_ab - g_rtf) / g_rtf
    assert rel < 1e-3


def test_age_based_curve_shape() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    P = build_kernel(mp)
    S = survival_from_kernel(P, mp, T_max=20_000)
    Ts = np.array([1, 10, 50, 100, 500])
    g = age_based_curve(S, Ts, cp.K_f, cp.K_p)
    assert g.shape == Ts.shape
    assert np.all(np.isfinite(g))
    assert np.all(g > 0.0)


def test_expected_tau_linear_system_matches_iterative() -> None:
    mp = baseline_model()
    P = build_kernel(mp)
    S = survival_from_kernel(P, mp, T_max=500_000, tol=1e-20)
    Etau_iter = expected_tau(S)
    Etau_lin = expected_tau_linear_system(P, mp)

    rel = abs(Etau_iter - Etau_lin) / Etau_lin
    assert rel < 1e-6


def test_g_rtf_iter_vs_linear() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    P = build_kernel(mp)
    S = survival_from_kernel(P, mp, T_max=500_000, tol=1e-20)
    g_iter = g_run_to_failure(S, cp.K_f)
    g_lin = g_run_to_failure_linear_system(P, mp, cp.K_f)
    rel = abs(g_iter - g_lin) / g_lin
    assert rel < 1e-6


def test_linear_system_baseline_value_sane() -> None:
    """Sanity: E[tau] from linear system is positive and finite."""
    mp = baseline_model()
    P = build_kernel(mp)
    Etau = expected_tau_linear_system(P, mp)
    assert np.isfinite(Etau)
    assert Etau > 1.0


def test_survival_no_early_stop_when_tol_negative() -> None:
    """When tol < 0, the survival vector has length exactly T_max + 1."""
    mp = baseline_model()
    P = build_kernel(mp)
    T_max = 500
    S = survival_from_kernel(P, mp, T_max=T_max, tol=-1.0)
    assert len(S) == T_max + 1
