"""Tests for ig_beta.py: IG MLE, IG sampler, beta log-linear fit."""
from __future__ import annotations

import numpy as np
import pytest

from buffer_policy.ig_beta import (
    BetaPoint,
    beta_implied_from_taus,
    fit_beta_log_model,
    ig_mle_closed_form,
    sample_inverse_gaussian,
)


def test_ig_mle_recovers_synthetic_within_5pct() -> None:
    mu_true, eta_true = 100.0, 50.0
    x = sample_inverse_gaussian(mu_true, eta_true, n=200_000, seed=2024)
    mu_hat, eta_hat = ig_mle_closed_form(x)
    assert abs(mu_hat - mu_true) / mu_true < 0.05
    assert abs(eta_hat - eta_true) / eta_true < 0.05


def test_ig_mle_rejects_nonpositive_data() -> None:
    with pytest.raises(ValueError):
        ig_mle_closed_form(np.array([1.0, 2.0, 0.0]))
    with pytest.raises(ValueError):
        ig_mle_closed_form(np.array([1.0, -2.0, 3.0]))
    with pytest.raises(ValueError):
        ig_mle_closed_form(np.array([]))


def test_ig_mle_rejects_constant_data() -> None:
    with pytest.raises(ValueError):
        ig_mle_closed_form(np.array([5.0, 5.0, 5.0, 5.0, 5.0]))


def test_ig_sampler_basic_stats() -> None:
    mu, eta = 50.0, 100.0
    x = sample_inverse_gaussian(mu, eta, n=100_000, seed=42)
    mean = float(x.mean())
    var = float(x.var())
    assert abs(mean - mu) / mu < 0.02
    assert abs(var - mu ** 3 / eta) / (mu ** 3 / eta) < 0.05


def _synthetic_beta_points(
    a: float, b: float, c: float,
    p_grid: list[float], delta_grid: list[float],
    C: int = 100, X0: int = 50,
    n_runs: int = 15000, n_hits_each: int = 14000,
) -> list[BetaPoint]:
    points = []
    for p in p_grid:
        for d in delta_grid:
            beta = a * (p ** (-b)) * np.exp(-c * d * np.log(p))
            lam = (1 - p) * C - d
            sigma_eff = lam + beta * p * (1 - p) * C ** 2
            eta = X0 ** 2 / sigma_eff
            points.append(BetaPoint(
                p=p, delta=d, lam=lam, C=C, X0=X0,
                n_runs=n_runs, n_hits=n_hits_each,
                no_hit_rate=1.0 - n_hits_each / n_runs,
                mu_hat=1.0, eta_hat=eta,
                sigma_eff_implied=sigma_eff,
                beta_implied=beta,
            ))
    return points


def test_beta_log_fit_recovers_synthetic_within_10pct() -> None:
    a_true, b_true, c_true = 0.8, 1.2, 0.05
    pts = _synthetic_beta_points(
        a_true, b_true, c_true,
        p_grid=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        delta_grid=[1.0, 3.0, 5.0, 8.0, 12.0, 15.0],
    )
    fit = fit_beta_log_model(pts, min_hits=500)
    assert abs(fit.a - a_true) / a_true < 0.10
    assert abs(fit.b - b_true) / b_true < 0.10
    assert abs(fit.c - c_true) / max(abs(c_true), 1e-12) < 0.10
    assert fit.r2_log > 0.99
    assert fit.r2_raw > 0.99
    assert fit.n_points_used == fit.n_points_total
    assert fit.n_discard_beta_nonpositive == 0
    assert fit.n_discard_low_hits == 0


def test_beta_log_fit_filters_low_hits_and_neg() -> None:
    a_true, b_true, c_true = 0.8, 1.2, 0.05
    pts = _synthetic_beta_points(
        a_true, b_true, c_true,
        p_grid=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        delta_grid=[1.0, 3.0, 5.0, 8.0, 12.0, 15.0],
    )
    pts.append(BetaPoint(
        p=0.5, delta=10.0, lam=40.0, C=100, X0=50,
        n_runs=15000, n_hits=100, no_hit_rate=0.99,
        mu_hat=1.0, eta_hat=1.0, sigma_eff_implied=1.0, beta_implied=0.5,
    ))
    pts.append(BetaPoint(
        p=0.4, delta=5.0, lam=55.0, C=100, X0=50,
        n_runs=15000, n_hits=14000, no_hit_rate=0.0,
        mu_hat=1.0, eta_hat=1.0, sigma_eff_implied=1.0, beta_implied=-0.2,
    ))
    fit = fit_beta_log_model(pts, min_hits=500)
    assert fit.n_discard_low_hits >= 1
    assert fit.n_discard_beta_nonpositive >= 1


def test_beta_implied_from_taus_basic() -> None:
    """Smoke check using X0=500 to avoid heavy-tailed sample."""
    p, delta, C, X0 = 0.15, 5.0, 100, 500
    lam = (1 - p) * C - delta
    sigma_eff = lam + 1.0 * p * (1 - p) * C ** 2
    eta_true = X0 ** 2 / sigma_eff
    mu_true = 200.0

    taus = sample_inverse_gaussian(mu_true, eta_true, n=100_000, seed=1)
    bp = beta_implied_from_taus(
        taus, n_runs=100_000, p=p, delta=delta, C=C, X0=X0,
    )
    assert abs(bp.beta_implied - 1.0) < 0.05
    assert bp.n_hits == 100_000
    assert bp.no_hit_rate == 0.0


def test_beta_implied_rejects_invalid_domain() -> None:
    taus = np.array([10.0, 12.0, 15.0, 20.0])
    with pytest.raises(ValueError):
        beta_implied_from_taus(taus, n_runs=4, p=1.2, delta=5.0, C=100, X0=50)
    with pytest.raises(ValueError):
        beta_implied_from_taus(taus, n_runs=4, p=0.2, delta=-1.0, C=100, X0=50)
    with pytest.raises(ValueError):
        beta_implied_from_taus(taus, n_runs=4, p=0.9, delta=50.0, C=10, X0=50)
    with pytest.raises(ValueError):
        beta_implied_from_taus(taus, n_runs=2, p=0.2, delta=5.0, C=100, X0=50)
