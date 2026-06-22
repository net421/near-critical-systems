"""
Inverse Gaussian (Wald) MLE and beta-correction calibration (modular).

Closed-form MLE:
    mu_hat   = (1/n) sum_i t_i
    1/eta_hat = (1/n) sum_i 1/t_i  -  1/mu_hat

Beta correction:
    sigma_eff^2_implied = X0^2 / eta_hat
    beta_implied         = (sigma_eff^2_implied - lambda) / (p (1-p) C^2)

Log-linear OLS fit:
    log(beta) = log(a) - b log(p) - c delta log(p)

The paper-grade NLS curve_fit with bounds lives in `ig_beta_paper.py` (Part 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def ig_mle_closed_form(tau: np.ndarray) -> tuple[float, float]:
    """Closed-form MLE for IG(mu, eta).  Returns (mu_hat, eta_hat)."""
    tau = np.asarray(tau, dtype=np.float64)
    if tau.size == 0:
        raise ValueError("tau is empty")
    if np.any(tau <= 0.0):
        raise ValueError("tau must be strictly positive")
    mu_hat = float(np.mean(tau))
    inv_mean = float(np.mean(1.0 / tau))
    gap = inv_mean - 1.0 / mu_hat
    if gap <= 0.0:
        raise ValueError(
            f"harmonic-vs-arithmetic gap = {gap:.3e} is non-positive; "
            "MLE undefined for this sample"
        )
    eta_hat = 1.0 / gap
    return mu_hat, eta_hat


def sample_inverse_gaussian(
    mu: float, eta: float, n: int, seed: int
) -> np.ndarray:
    """Sample n i.i.d. IG(mu, eta) variates via Michael-Schucany-Haas (1976)."""
    if mu <= 0.0 or eta <= 0.0:
        raise ValueError("mu, eta must be > 0")
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.default_rng(seed)
    V = rng.standard_normal(n)
    Y = V * V
    X = (
        mu
        + (mu * mu * Y) / (2.0 * eta)
        - (mu / (2.0 * eta))
          * np.sqrt(4.0 * mu * eta * Y + mu * mu * Y * Y)
    )
    U = rng.random(n)
    threshold = mu / (mu + X)
    return np.where(U <= threshold, X, (mu * mu) / X)


@dataclass(frozen=True)
class BetaPoint:
    p: float
    delta: float
    lam: float
    C: int
    X0: int
    n_runs: int
    n_hits: int
    no_hit_rate: float
    mu_hat: float
    eta_hat: float
    sigma_eff_implied: float
    beta_implied: float


def beta_implied_from_taus(
    taus_observed: np.ndarray,
    n_runs: int,
    p: float,
    delta: float,
    C: int,
    X0: int,
) -> BetaPoint:
    """Build a BetaPoint from observed (uncensored) hit times."""
    taus_observed = np.asarray(taus_observed, dtype=np.float64)

    if n_runs < 1:
        raise ValueError(f"n_runs must be >= 1; got {n_runs}")
    if taus_observed.size > n_runs:
        raise ValueError(
            f"n_hits ({taus_observed.size}) cannot exceed n_runs ({n_runs})"
        )
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1); got {p}")
    if delta <= 0.0:
        raise ValueError(f"delta must be > 0; got {delta}")
    if C <= 0:
        raise ValueError(f"C must be > 0; got {C}")
    if X0 <= 0:
        raise ValueError(f"X0 must be > 0; got {X0}")

    lam = (1.0 - p) * C - delta
    if lam <= 0.0:
        raise ValueError(
            f"implied lambda must be > 0; got lam={lam}"
        )

    n_hits = int(taus_observed.size)
    no_hit_rate = float(1.0 - n_hits / max(1, n_runs))

    mu_hat, eta_hat = ig_mle_closed_form(taus_observed)
    sigma_eff_implied = (X0 ** 2) / eta_hat
    denom = p * (1.0 - p) * C * C
    beta_implied = (sigma_eff_implied - lam) / denom

    return BetaPoint(
        p=float(p), delta=float(delta), lam=float(lam),
        C=int(C), X0=int(X0),
        n_runs=int(n_runs), n_hits=n_hits,
        no_hit_rate=no_hit_rate,
        mu_hat=float(mu_hat), eta_hat=float(eta_hat),
        sigma_eff_implied=float(sigma_eff_implied),
        beta_implied=float(beta_implied),
    )


@dataclass(frozen=True)
class BetaFitResult:
    a: float
    b: float
    c: float
    se_A: float
    se_a: float
    se_b: float
    se_c: float
    r2_log: float
    r2_raw: float
    mean_rel_error_eta: float
    max_rel_error_eta: float
    n_points_used: int
    n_points_total: int
    n_discard_beta_nonpositive: int
    n_discard_low_hits: int


def fit_beta_log_model(
    points: list[BetaPoint],
    min_hits: int = 500,
) -> BetaFitResult:
    """Fit log(beta) = log(a) - b log(p) - c delta log(p) via OLS."""
    n_total = len(points)
    if n_total < 4:
        raise ValueError(f"need at least 4 points; got {n_total}")

    discard_low_hits = sum(1 for pt in points if pt.n_hits < min_hits)
    after_hits = [pt for pt in points if pt.n_hits >= min_hits]
    discard_neg = sum(1 for pt in after_hits if pt.beta_implied <= 0.0)
    used = [pt for pt in after_hits if pt.beta_implied > 0.0]

    n_used = len(used)
    if n_used < 4:
        raise ValueError(
            f"after filtering only {n_used} points remain; need >= 4"
        )

    p_arr = np.array([pt.p for pt in used], dtype=np.float64)
    d_arr = np.array([pt.delta for pt in used], dtype=np.float64)
    beta_arr = np.array([pt.beta_implied for pt in used], dtype=np.float64)
    eta_arr = np.array([pt.eta_hat for pt in used], dtype=np.float64)
    lam_arr = np.array([pt.lam for pt in used], dtype=np.float64)
    C_arr = np.array([pt.C for pt in used], dtype=np.float64)
    X0_arr = np.array([pt.X0 for pt in used], dtype=np.float64)

    log_p = np.log(p_arr)
    y = np.log(beta_arr)
    x1 = -log_p
    x2 = -d_arr * log_p
    X = np.column_stack([np.ones_like(y), x1, x2])

    theta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    A, B, Cc = float(theta[0]), float(theta[1]), float(theta[2])
    a = float(np.exp(A))
    b = B
    c = Cc

    y_hat = X @ theta
    resid = y - y_hat
    n, k = X.shape
    sigma2 = float(resid @ resid / (n - k)) if n > k else float("nan")
    XtX_inv = np.linalg.inv(X.T @ X)
    var_theta = sigma2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(var_theta), 0.0))
    se_A, se_B, se_C = float(se[0]), float(se[1]), float(se[2])
    se_a = a * se_A

    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2_log = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")

    beta_hat_raw = np.exp(y_hat)
    ss_res_raw = float(((beta_arr - beta_hat_raw) ** 2).sum())
    ss_tot_raw = float(((beta_arr - beta_arr.mean()) ** 2).sum())
    r2_raw = (1.0 - ss_res_raw / ss_tot_raw
              if ss_tot_raw > 0.0 else float("nan"))

    sigma_eff_model = lam_arr + beta_hat_raw * p_arr * (1.0 - p_arr) * C_arr ** 2
    eta_model = (X0_arr ** 2) / sigma_eff_model
    rel_err_eta = np.abs(eta_model - eta_arr) / eta_arr
    mean_rel_eta = float(rel_err_eta.mean())
    max_rel_eta = float(rel_err_eta.max())

    return BetaFitResult(
        a=a, b=b, c=c,
        se_A=se_A, se_a=se_a, se_b=se_B, se_c=se_C,
        r2_log=r2_log, r2_raw=r2_raw,
        mean_rel_error_eta=mean_rel_eta,
        max_rel_error_eta=max_rel_eta,
        n_points_used=n_used,
        n_points_total=n_total,
        n_discard_beta_nonpositive=discard_neg,
        n_discard_low_hits=discard_low_hits,
    )
