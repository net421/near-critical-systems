"""
Parameter bundles for the inventory model.

Model recap
-----------
    I(t+1) = min{ I(t) + Y(t) - D(t), S_max }
    Y(t) ~ C w.p. 1-p, 0 w.p. p
    D(t) ~ Poisson(lambda)

Stability:   delta   = (1-p) * C - lambda > 0
Variance:    sigma^2 = p*(1-p)*C^2 + lambda
Utilization: u       = lambda / C
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelParams:
    """Physical parameters of the inventory system."""
    p: float = 0.15
    C: int = 100
    lam: float = 80.0
    S_max: int = 180
    X0: int = 50
    s_reset: int = 50

    def __post_init__(self) -> None:
        if not (0.0 < self.p < 1.0):
            raise ValueError(f"p must be in (0,1); got {self.p}")
        if self.C <= 0:
            raise ValueError(f"C must be > 0; got {self.C}")
        if self.lam < 0:
            raise ValueError(f"lambda must be >= 0; got {self.lam}")
        if self.S_max < 1:
            raise ValueError(f"S_max must be >= 1; got {self.S_max}")
        if not (1 <= self.X0 <= self.S_max):
            raise ValueError(
                f"X0 must lie in [1, S_max]; got X0={self.X0}, "
                f"S_max={self.S_max}"
            )
        if not (1 <= self.s_reset <= self.S_max):
            raise ValueError(
                f"s_reset must lie in [1, S_max]; got {self.s_reset}"
            )
        if self.delta <= 0.0:
            raise ValueError(
                f"Instability: delta = (1-p)C - lambda = {self.delta:.4g} <= 0"
            )

    @property
    def delta(self) -> float:
        """Stability margin delta = (1-p) C - lambda."""
        return (1.0 - self.p) * self.C - self.lam

    @property
    def sigma2(self) -> float:
        """Per-period variance sigma^2 = p(1-p) C^2 + lambda."""
        return self.p * (1.0 - self.p) * self.C ** 2 + self.lam

    @property
    def utilization(self) -> float:
        """Utilization u = lambda / C."""
        return self.lam / self.C


@dataclass(frozen=True)
class CostParams:
    """Cost structure: failure cost K_f, preventive cost K_p."""
    K_f: float = 100.0
    K_p: float = 10.0

    def __post_init__(self) -> None:
        if self.K_f <= 0:
            raise ValueError(f"K_f must be > 0; got {self.K_f}")
        if self.K_p <= 0:
            raise ValueError(f"K_p must be > 0; got {self.K_p}")

    @property
    def ratio(self) -> float:
        """Cost ratio K_f / K_p."""
        return self.K_f / self.K_p


def baseline_model() -> ModelParams:
    """Baseline parameters: p=0.15, C=100, lambda=80, S_max=180, X0=s_reset=50."""
    return ModelParams()


def baseline_cost() -> CostParams:
    """Baseline costs: K_f=100, K_p=10."""
    return CostParams()
