"""Tests for hazard.py: discrete hazard and DFR diagnostics."""
from __future__ import annotations

import numpy as np

from buffer_policy.hazard import (
    discrete_hazard,
    hazard_diagnostics,
    moving_average,
)


def test_discrete_hazard_basic_shape() -> None:
    S = np.array([1.0, 0.8, 0.6, 0.5, 0.45])
    h = discrete_hazard(S)
    assert h.shape == (4,)
    np.testing.assert_allclose(h, [0.2, 0.25, 1/6, 0.1], atol=1e-12)


def test_discrete_hazard_filters_tail() -> None:
    S = np.array([1.0, 0.5, 1e-9, 1e-12])
    h = discrete_hazard(S, survival_tol=1e-8)
    assert h.shape == (2,)


def test_moving_average_basic() -> None:
    x = np.arange(10, dtype=float)
    m = moving_average(x, window=3)
    expected = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
    np.testing.assert_allclose(m, expected, atol=1e-12)


def test_dfr_strict_on_mixture_of_exponentials() -> None:
    """Mixture of exponentials has strictly decreasing hazard."""
    t = np.arange(0, 2000, dtype=float)
    S = 0.7 * np.exp(-0.005 * t) + 0.3 * np.exp(-0.05 * t)
    diag = hazard_diagnostics(S, window=25, survival_tol=1e-8)
    assert diag.DFR_strict_raw is True
    assert diag.DFR_strict_smoothed is True
    assert diag.DFR_approx_smoothed is True
    assert diag.frac_decreasing_smoothed >= 0.99
    assert diag.h_initial > diag.h_final
    assert diag.n_hazard_points > 100


def test_dfr_fails_on_increasing_hazard() -> None:
    h = np.linspace(0.01, 0.5, 200)
    S = [1.0]
    for hi in h:
        S.append(S[-1] * (1.0 - hi))
    S = np.asarray(S)
    diag = hazard_diagnostics(S, window=25, survival_tol=1e-8)
    assert diag.DFR_strict_raw is False
    assert diag.DFR_approx_smoothed is False


def test_hazard_diagnostics_short_input() -> None:
    S = np.array([1.0, 0.5])
    diag = hazard_diagnostics(S)
    assert diag.n_hazard_points == 1
    assert diag.DFR_strict_raw is False


def test_h_initial_corresponds_to_t_equals_one() -> None:
    """h_initial = h[0] = (S[0]-S[1])/S[0] is mathematically h(t=1)."""
    S = np.array([1.0, 0.8, 0.6, 0.5])
    diag = hazard_diagnostics(S)
    assert abs(diag.h_initial - 0.2) < 1e-12
