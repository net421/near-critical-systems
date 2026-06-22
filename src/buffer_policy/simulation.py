"""
Uncapped Monte Carlo first-passage simulator.

Model (UNCAPPED, NO S_max):
    I(t+1) = I(t) + Y(t) - D(t)
    Y(t)   = 0 with prob. p,  Y(t) = C with prob. 1-p
    D(t)   ~ Poisson(lambda)
    tau    = inf{ t >= 1 : I(t) <= 0 }

This simulator is used ONLY for the beta-calibration regime in Phase 3/4.
It is mathematically distinct from the capped MDP kernel of Phases 1-2.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit, prange


def make_child_seeds(seed: int, n: int) -> np.ndarray:
    """Spawn n independent uint32 seeds from a master seed via SeedSequence."""
    if n <= 0:
        raise ValueError("n must be > 0")
    ss = np.random.SeedSequence(seed)
    children = ss.spawn(n)
    return np.array(
        [c.generate_state(1, dtype=np.uint32)[0] for c in children],
        dtype=np.uint32,
    )


@njit(parallel=True, cache=True, fastmath=False)
def _simulate_uncapped_numba(
    X0, C, p, lam, n_runs, max_steps,
    seeds, taus_out, hit_out,
):
    for i in prange(n_runs):
        np.random.seed(int(seeds[i]))
        I = X0
        tau_i = max_steps
        hit_i = 0
        for t in range(1, max_steps + 1):
            U = np.random.random()
            Y = 0 if U < p else C
            D = np.random.poisson(lam)
            I = I + Y - D
            if I <= 0:
                tau_i = t
                hit_i = 1
                break
        taus_out[i] = tau_i
        hit_out[i] = hit_i


def _simulate_uncapped_python(X0, C, p, lam, n_runs, max_steps, seeds):
    taus = np.empty(n_runs, dtype=np.int64)
    hit = np.zeros(n_runs, dtype=bool)
    for i in range(n_runs):
        rng = np.random.default_rng(int(seeds[i]))
        I = X0
        tau_i = max_steps
        hit_i = False
        for t in range(1, max_steps + 1):
            U = rng.random()
            Y = 0 if U < p else C
            D = int(rng.poisson(lam))
            I = I + Y - D
            if I <= 0:
                tau_i = t
                hit_i = True
                break
        taus[i] = tau_i
        hit[i] = hit_i
    return taus, hit


@dataclass(frozen=True)
class FirstPassageResult:
    taus: np.ndarray
    hit: np.ndarray
    seed: int
    n_runs: int
    max_steps: int
    X0: int
    C: int
    p: float
    lam: float
    backend: str

    @property
    def n_hits(self) -> int:
        return int(self.hit.sum())

    @property
    def n_censored(self) -> int:
        return int(self.n_runs - self.n_hits)

    @property
    def no_hit_rate(self) -> float:
        return float(1.0 - self.hit.mean())

    @property
    def observed_taus(self) -> np.ndarray:
        return self.taus[self.hit]

    def summary(self) -> dict[str, Any]:
        return {
            "n_runs": self.n_runs,
            "n_hits": self.n_hits,
            "n_censored": self.n_censored,
            "no_hit_rate": self.no_hit_rate,
            "max_steps": self.max_steps,
            "seed": int(self.seed),
            "backend": self.backend,
            "params": {"X0": self.X0, "C": self.C,
                       "p": self.p, "lam": self.lam},
        }


def simulate_uncapped_first_passage(
    X0: int,
    C: int,
    p: float,
    lam: float,
    n_runs: int,
    max_steps: int,
    seed: int,
    use_numba: bool = True,
) -> FirstPassageResult:
    """Run n_runs independent uncapped trajectories until collapse or max_steps."""
    if X0 < 1:
        raise ValueError(f"X0 must be >= 1; got {X0}")
    if C < 1:
        raise ValueError(f"C must be >= 1; got {C}")
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1); got {p}")
    if lam <= 0.0:
        raise ValueError(f"lam must be > 0; got {lam}")
    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1; got {n_runs}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1; got {max_steps}")

    seeds = make_child_seeds(seed, n_runs)

    if use_numba:
        taus = np.empty(n_runs, dtype=np.int64)
        hit_u8 = np.empty(n_runs, dtype=np.uint8)
        _simulate_uncapped_numba(
            int(X0), int(C), float(p), float(lam),
            int(n_runs), int(max_steps), seeds, taus, hit_u8,
        )
        hit = hit_u8.astype(bool)
        backend = "numba"
    else:
        taus, hit = _simulate_uncapped_python(
            int(X0), int(C), float(p), float(lam),
            int(n_runs), int(max_steps), seeds,
        )
        backend = "python"

    return FirstPassageResult(
        taus=taus, hit=hit, seed=int(seed),
        n_runs=int(n_runs), max_steps=int(max_steps),
        X0=int(X0), C=int(C), p=float(p), lam=float(lam),
        backend=backend,
    )
