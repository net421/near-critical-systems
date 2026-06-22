"""Kernel-construction tests."""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from buffer_policy.kernel import build_kernel, q_fail_vector
from buffer_policy.params import baseline_model


def test_row_sums_baseline() -> None:
    mp = baseline_model()
    P = build_kernel(mp)
    rs = P.sum(axis=1)
    assert np.max(np.abs(rs - 1.0)) < 1e-12


def test_state_zero_absorbing() -> None:
    mp = baseline_model()
    P = build_kernel(mp)
    assert P[0, 0] == 1.0
    assert P[0, 1:].sum() == 0.0


def test_row_at_s90_matches_scipy() -> None:
    """Reconstruct P[90, :] from scratch via scipy and compare."""
    mp = baseline_model()
    P = build_kernel(mp)
    s = 90
    p, C, lam, S = mp.p, mp.C, mp.lam, mp.S_max

    expected = np.zeros(S + 1)
    expected[0] = (
        (1 - p) * poisson.sf(s + C - 1, mu=lam)
        + p * poisson.sf(s - 1, mu=lam)
    )
    for sp in range(1, s + 1):
        expected[sp] += p * poisson.pmf(s - sp, mu=lam)
    for sp in range(1, S):
        expected[sp] += (1 - p) * poisson.pmf(s + C - sp, mu=lam)
    m = s + C - S
    if m >= 0:
        expected[S] += (1 - p) * poisson.cdf(m, mu=lam)

    assert np.max(np.abs(P[s, :] - expected)) < 1e-14


def test_q_fail_vector_consistency() -> None:
    mp = baseline_model()
    P = build_kernel(mp)
    q = q_fail_vector(mp)
    assert np.max(np.abs(P[1:, 0] - q[1:])) < 1e-14


def test_q_fail_vector_state_zero() -> None:
    mp = baseline_model()
    q = q_fail_vector(mp)
    assert q[0] == 1.0


def test_no_negative_probabilities() -> None:
    mp = baseline_model()
    P = build_kernel(mp)
    assert np.all(P >= 0.0)


def test_row_at_low_state() -> None:
    """Row at s=1 is a tight stress case."""
    mp = baseline_model()
    P = build_kernel(mp)
    rs = P[1].sum()
    assert abs(rs - 1.0) < 1e-13
