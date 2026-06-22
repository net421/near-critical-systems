"""Cross-cutting reproducibility tests for the seed pipeline and simulator."""
from __future__ import annotations

import numpy as np
import pytest

from buffer_policy.simulation import (
    make_child_seeds,
    simulate_uncapped_first_passage,
)


def test_seed_pipeline_is_deterministic() -> None:
    for s in (0, 1, 42, 2024, 99999):
        a = make_child_seeds(s, 50)
        b = make_child_seeds(s, 50)
        assert np.array_equal(a, b), f"seed={s} not deterministic"


def test_seed_pipeline_distinct_for_different_master_seeds() -> None:
    a = make_child_seeds(1, 1000)
    b = make_child_seeds(2, 1000)
    overlap = int(np.sum(a == b))
    assert overlap < 5, f"too many coincidences: {overlap}"


@pytest.mark.parametrize("use_numba", [True, False])
def test_simulator_full_reproducibility_round_trip(use_numba: bool) -> None:
    kw = dict(X0=15, C=30, p=0.18, lam=22.0,
              n_runs=300, max_steps=3000, seed=12345,
              use_numba=use_numba)
    r1 = simulate_uncapped_first_passage(**kw)
    r2 = simulate_uncapped_first_passage(**kw)
    assert np.array_equal(r1.taus, r2.taus)
    assert np.array_equal(r1.hit, r2.hit)
    assert r1.n_hits == r2.n_hits
    assert r1.no_hit_rate == r2.no_hit_rate
    assert r1.backend == r2.backend


@pytest.mark.parametrize("use_numba", [True, False])
def test_simulator_changing_only_seed_changes_output(use_numba: bool) -> None:
    base = dict(X0=15, C=30, p=0.18, lam=22.0,
                n_runs=300, max_steps=3000, use_numba=use_numba)
    r1 = simulate_uncapped_first_passage(**base, seed=12345)
    r2 = simulate_uncapped_first_passage(**base, seed=54321)
    different = (
        not np.array_equal(r1.taus, r2.taus)
        or not np.array_equal(r1.hit, r2.hit)
    )
    assert different
