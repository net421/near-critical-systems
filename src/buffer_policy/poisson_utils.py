"""
Numerically stable Poisson PMF, CDF, and tail probabilities.

We use scipy.stats.poisson for reference-quality results.  The kernel
construction calls these helpers; they must agree with scipy to ~1e-15.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson


def poisson_pmf(k, lam: float) -> np.ndarray:
    """P(D = k) for D ~ Poisson(lam).  Supports vector k."""
    k = np.asarray(k)
    return poisson.pmf(k, mu=lam)


def poisson_cdf(k, lam: float) -> np.ndarray:
    """P(D <= k)."""
    k = np.asarray(k)
    return poisson.cdf(k, mu=lam)


def poisson_sf(k, lam: float) -> np.ndarray:
    """Survival P(D > k) = 1 - CDF(k).  Hence P(D >= n) = sf(n-1)."""
    k = np.asarray(k)
    return poisson.sf(k, mu=lam)


def poisson_tail_ge(n, lam: float) -> np.ndarray:
    """P(D >= n) = sf(n-1).  Vector-safe."""
    n = np.asarray(n)
    return poisson.sf(n - 1, mu=lam)


def truncated_pmf_array(lam: float, n_max: int) -> np.ndarray:
    """
    Return array `pmf` of length `n_max + 1` with pmf[k] = P(D = k) for
    0 <= k <= n_max.  Used for fast kernel assembly.
    """
    if n_max < 0:
        raise ValueError("n_max must be >= 0")
    return poisson.pmf(np.arange(n_max + 1), mu=lam)
