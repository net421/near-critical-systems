"""Tests for poisson_utils against scipy reference."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson

from buffer_policy.poisson_utils import (
    poisson_cdf,
    poisson_pmf,
    poisson_sf,
    poisson_tail_ge,
    truncated_pmf_array,
)


@pytest.mark.parametrize("lam", [1.0, 5.0, 80.0, 200.0])
def test_pmf_matches_scipy(lam: float) -> None:
    k = np.arange(0, int(lam) * 3 + 30)
    assert np.allclose(poisson_pmf(k, lam), poisson.pmf(k, mu=lam), atol=1e-15)


def test_tail_ge_identity() -> None:
    lam = 80.0
    n = np.arange(1, 200)
    assert np.allclose(
        poisson_tail_ge(n, lam), poisson.sf(n - 1, mu=lam), atol=1e-15
    )


def test_cdf_sf_complement() -> None:
    lam = 80.0
    k = np.arange(0, 200)
    assert np.allclose(poisson_cdf(k, lam) + poisson_sf(k, lam), 1.0, atol=1e-14)


def test_truncated_pmf_array_sums_to_almost_one() -> None:
    lam = 80.0
    pmf = truncated_pmf_array(lam, 250)
    assert pmf.shape == (251,)
    assert abs(pmf.sum() - 1.0) < 1e-12
