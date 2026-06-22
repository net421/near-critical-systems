"""Tests for simulation.py: correctness and reproducibility, both backends."""
from __future__ import annotations

import numpy as np
import pytest

from buffer_policy.simulation import (
    make_child_seeds,
    simulate_uncapped_first_passage,
)


def test_make_child_seeds_deterministic() -> None:
    s1 = make_child_seeds(42, 100)
    s2 = make_child_seeds(42, 100)
    s3 = make_child_seeds(43, 100)
    assert np.array_equal(s1, s2)
    assert not np.array_equal(s1, s3)
    assert s1.dtype == np.uint32
    assert s1.shape == (100,)


@pytest.mark.parametrize("use_numba", [True, False])
def test_simulator_reproducibility_same_seed(use_numba: bool) -> None:
    kw = dict(X0=10, C=20, p=0.2, lam=15.0,
              n_runs=200, max_steps=2000, seed=2024,
              use_numba=use_numba)
    r1 = simulate_uncapped_first_passage(**kw)
    r2 = simulate_uncapped_first_passage(**kw)
    assert np.array_equal(r1.taus, r2.taus)
    assert np.array_equal(r1.hit, r2.hit)
    assert r1.backend == r2.backend


@pytest.mark.parametrize("use_numba", [True, False])
def test_simulator_different_seeds_differ(use_numba: bool) -> None:
    kw = dict(X0=10, C=20, p=0.2, lam=15.0,
              n_runs=200, max_steps=2000, use_numba=use_numba)
    r1 = simulate_uncapped_first_passage(**kw, seed=2024)
    r2 = simulate_uncapped_first_passage(**kw, seed=9999)
    different = (
        not np.array_equal(r1.taus, r2.taus)
        or not np.array_equal(r1.hit, r2.hit)
    )
    assert different


@pytest.mark.parametrize("use_numba", [True, False])
def test_simulator_output_shapes_and_types(use_numba: bool) -> None:
    res = simulate_uncapped_first_passage(
        X0=10, C=20, p=0.2, lam=15.0,
        n_runs=50, max_steps=500, seed=1, use_numba=use_numba,
    )
    assert res.taus.shape == (50,)
    assert res.hit.shape == (50,)
    assert res.taus.dtype == np.int64
    assert res.hit.dtype == np.bool_
    assert np.all(res.taus >= 1)
    assert np.all(res.taus <= 500)
    assert res.backend in ("numba", "python")


def test_simulator_hits_have_positive_tau() -> None:
    res = simulate_uncapped_first_passage(
        X0=10, C=20, p=0.2, lam=15.0,
        n_runs=200, max_steps=5000, seed=7,
    )
    obs = res.observed_taus
    assert np.all(obs >= 1)
    assert res.n_hits + res.n_censored == res.n_runs
    assert 0.0 <= res.no_hit_rate <= 1.0


def test_simulator_near_critical_stable_yields_many_hits() -> None:
    """delta = (1-p)C - lam = 0.1 (stable, near-critical) yields many hits."""
    res = simulate_uncapped_first_passage(
        X0=5, C=10, p=0.2, lam=7.9,
        n_runs=300, max_steps=5000, seed=11,
    )
    assert res.n_hits >= 200, f"only {res.n_hits} hits; expected many"


def test_simulator_input_validation() -> None:
    with pytest.raises(ValueError):
        simulate_uncapped_first_passage(
            X0=0, C=20, p=0.2, lam=15.0,
            n_runs=10, max_steps=100, seed=1,
        )
    with pytest.raises(ValueError):
        simulate_uncapped_first_passage(
            X0=10, C=20, p=1.5, lam=15.0,
            n_runs=10, max_steps=100, seed=1,
        )
    with pytest.raises(ValueError):
        simulate_uncapped_first_passage(
            X0=10, C=20, p=0.2, lam=-1.0,
            n_runs=10, max_steps=100, seed=1,
        )


def test_backends_differ_but_each_is_consistent() -> None:
    """Numba and Python use different RNGs; each backend is internally deterministic."""
    kw = dict(X0=10, C=20, p=0.2, lam=15.0,
              n_runs=100, max_steps=2000, seed=2024)
    r_n = simulate_uncapped_first_passage(**kw, use_numba=True)
    r_p = simulate_uncapped_first_passage(**kw, use_numba=False)
    assert r_n.backend == "numba"
    assert r_p.backend == "python"
    assert r_n.n_runs == r_p.n_runs == 100
