"""
Paper-grade NLS beta calibration.

Model:
    beta(p, delta) = a * p^(-(b + c*delta))

Equivalent log form:
    log(beta) = log(a) - (b + c*delta) * log(p)

Fit via scipy.optimize.curve_fit with bounds:
    p0     = (0.24, 0.41, 0.011)
    bounds = ([1e-4, 1e-4, -1.0], [10.0, 5.0, 1.0])

Reports both raw-beta and eta-level error metrics, plus subgroup
breakdowns for paper Table 3:
    - error subgroups by p:     (0, 0.10], (0.10, 0.20], (0.20, 0.35]
    - error subgroups by delta: (0, 3], (3, 7], (7, 10], (10, 15]

The pipeline `run_beta_calibration_paper` orchestrates:
    1. Generate two grids (expanded + delta_fixed) and combine them.
    2. Spawn child seeds (SeedSequence.spawn or legacy seed+i*1000).
    3. Run uncapped Monte Carlo per design point with Numba fallback.
    4. Filter on n_collapsed >= min_hits, dedupe by (p, lam) rounded
       to 4 decimals, then apply validity filter
       p <= p_max_validity, delta <= delta_max_validity
       => 42 design points in the canonical paper config.
    5. Build BetaPoint records and fit `fit_beta_nls_paper`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from buffer_policy.ig_beta import BetaPoint, beta_implied_from_taus
from buffer_policy.simulation import simulate_uncapped_first_passage


# --------------------------------------------------------------- model
def beta_model(X, a, b, c):
    """beta(p, delta) = a * p^(-(b + c*delta))."""
    p, delta = X
    return a * np.power(p, -(b + c * delta))


# --------------------------------------------------------------- result
@dataclass(frozen=True)
class BetaFitPaperResult:
    a: float
    b: float
    c: float
    R2_raw_beta: float
    n_points_used: int
    n_points_total: int
    # eta-level metrics
    MAE_uncorrected_eta_pct: float
    MAE_corrected_eta_pct: float
    max_error_uncorrected_eta_pct: float
    max_error_corrected_eta_pct: float
    median_error_uncorrected_eta_pct: float
    median_error_corrected_eta_pct: float
    improvement_factor: float
    # subgroup breakdowns
    err_eta_pct_by_p: dict = field(default_factory=dict)
    err_eta_pct_by_delta: dict = field(default_factory=dict)
    cov_diag: list = field(default_factory=list)


# --------------------------------------------------------------- subgroup helper
def _subgroup_stats(
    bins: list[tuple[float, float]],
    values: np.ndarray,
    errors: np.ndarray,
) -> dict:
    out = {}
    for lo, hi in bins:
        mask = (values > lo) & (values <= hi)
        n = int(mask.sum())
        if n == 0:
            out[(lo, hi)] = {"n": 0, "mean": float("nan"),
                             "max": float("nan")}
        else:
            sub = errors[mask]
            out[(lo, hi)] = {
                "n": n,
                "mean": float(sub.mean()),
                "max": float(sub.max()),
            }
    return out


# --------------------------------------------------------------- fit
def fit_beta_nls_paper(
    points: list[BetaPoint],
    min_hits: int = 500,
) -> BetaFitPaperResult:
    """
    Non-linear least squares fit of beta(p,delta) = a*p^(-(b+c*delta))
    in raw beta space (NOT log space), with parameter bounds.
    """
    n_total = len(points)
    used = [pt for pt in points
            if pt.n_hits >= min_hits and pt.beta_implied > 0.0]
    n_used = len(used)
    if n_used < 4:
        raise ValueError(
            f"need at least 4 valid points; got {n_used} of {n_total}"
        )

    p_arr = np.array([pt.p for pt in used], dtype=np.float64)
    d_arr = np.array([pt.delta for pt in used], dtype=np.float64)
    beta_arr = np.array([pt.beta_implied for pt in used], dtype=np.float64)
    eta_arr = np.array([pt.eta_hat for pt in used], dtype=np.float64)
    lam_arr = np.array([pt.lam for pt in used], dtype=np.float64)
    C_arr = np.array([pt.C for pt in used], dtype=np.float64)
    X0_arr = np.array([pt.X0 for pt in used], dtype=np.float64)

    p0 = (0.24, 0.41, 0.011)
    bounds = ([1e-4, 1e-4, -1.0], [10.0, 5.0, 1.0])

    popt, pcov = curve_fit(
        beta_model,
        (p_arr, d_arr),
        beta_arr,
        p0=p0,
        bounds=bounds,
        maxfev=20_000,
    )
    a, b, c = float(popt[0]), float(popt[1]), float(popt[2])
    cov_diag = [float(v) for v in np.diag(pcov)]

    # raw-beta R^2
    beta_pred = beta_model((p_arr, d_arr), a, b, c)
    ss_res = float(((beta_arr - beta_pred) ** 2).sum())
    ss_tot = float(((beta_arr - beta_arr.mean()) ** 2).sum())
    r2_raw = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")

    # ---- uncorrected eta (assumes beta=1 implicitly)
    sigma2_unc = lam_arr + p_arr * (1.0 - p_arr) * C_arr ** 2
    eta_unc = (X0_arr ** 2) / sigma2_unc
    err_unc_pct = 100.0 * np.abs(eta_unc - eta_arr) / eta_arr

    # ---- corrected eta (using fitted beta)
    sigma2_corr = lam_arr + beta_pred * p_arr * (1.0 - p_arr) * C_arr ** 2
    eta_corr = (X0_arr ** 2) / sigma2_corr
    err_corr_pct = 100.0 * np.abs(eta_corr - eta_arr) / eta_arr

    mae_unc = float(err_unc_pct.mean())
    mae_corr = float(err_corr_pct.mean())
    improvement = (mae_unc / mae_corr) if mae_corr > 0.0 else float("inf")

    # ---- subgroup breakdowns
    p_bins = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.35)]
    delta_bins = [(0.0, 3.0), (3.0, 7.0), (7.0, 10.0), (10.0, 15.0)]
    err_by_p = _subgroup_stats(p_bins, p_arr, err_corr_pct)
    err_by_delta = _subgroup_stats(delta_bins, d_arr, err_corr_pct)

    return BetaFitPaperResult(
        a=a, b=b, c=c,
        R2_raw_beta=r2_raw,
        n_points_used=n_used,
        n_points_total=n_total,
        MAE_uncorrected_eta_pct=mae_unc,
        MAE_corrected_eta_pct=mae_corr,
        max_error_uncorrected_eta_pct=float(err_unc_pct.max()),
        max_error_corrected_eta_pct=float(err_corr_pct.max()),
        median_error_uncorrected_eta_pct=float(np.median(err_unc_pct)),
        median_error_corrected_eta_pct=float(np.median(err_corr_pct)),
        improvement_factor=float(improvement),
        err_eta_pct_by_p=err_by_p,
        err_eta_pct_by_delta=err_by_delta,
        cov_diag=cov_diag,
    )


# --------------------------------------------------------------- grid generators
def generate_grid_expanded_paper(C: int = 100) -> list[dict]:
    """Cartesian (p, util) filtered by delta in [0.5, 20], lam in (0, C)."""
    p_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30,
              0.35, 0.40, 0.45, 0.50]
    util_list = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    out: list[dict] = []
    for p in p_list:
        for u in util_list:
            lam = u * C
            if not (0.0 < lam < C):
                continue
            delta = (1.0 - p) * C - lam
            if not (0.5 <= delta <= 20.0):
                continue
            out.append({
                "p": float(p), "lam": float(lam), "delta": float(delta),
                "C": int(C), "source": "expanded",
            })
    return out


def generate_grid_delta_fixed_paper(C: int = 100) -> list[dict]:
    """Fixed delta in {2, 5, 10}, p in {0.05, ..., 0.50}, lam in (0, C)."""
    deltas = [2.0, 5.0, 10.0]
    p_list = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
              0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    out: list[dict] = []
    for p in p_list:
        for delta in deltas:
            lam = (1.0 - p) * C - delta
            if not (0.0 < lam < C):
                continue
            out.append({
                "p": float(p), "lam": float(lam), "delta": float(delta),
                "C": int(C), "source": "delta_fixed",
            })
    return out


# --------------------------------------------------------------- pipeline
def run_beta_calibration_paper(
    n_runs: int,
    max_steps: int,
    seed: int,
    seed_mode: str = "spawn",
    use_numba: bool = True,
    min_hits: int = 500,
    p_max_validity: float = 0.35,
    delta_max_validity: float = 15.0,
    C: int = 100,
    X0: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Full paper-grade beta calibration pipeline.

    Parameters
    ----------
    seed_mode : "spawn" or "legacy_seed_i"
        "spawn"          : SeedSequence.spawn(n_configs).
        "legacy_seed_i"  : seed_cfg = seed + idx * 1000.

    Returns
    -------
    df_all_points : DataFrame with one row per design point (all configs).
    df_validity_filtered : subset with p<=p_max_validity, delta<=delta_max_validity.
    fit_summary : dict (asdict of BetaFitPaperResult), or {"error": ...}.
    """
    if seed_mode not in ("spawn", "legacy_seed_i"):
        raise ValueError(f"unknown seed_mode: {seed_mode}")

    grid_a = generate_grid_expanded_paper(C=C)
    grid_b = generate_grid_delta_fixed_paper(C=C)
    configs = grid_a + grid_b
    n_configs = len(configs)

    if seed_mode == "spawn":
        ss = np.random.SeedSequence(seed)
        children = ss.spawn(n_configs)
        seeds = [int(c.generate_state(1, dtype=np.uint32)[0])
                 for c in children]
    else:  # legacy_seed_i
        seeds = [int(seed + idx * 1000) for idx in range(n_configs)]

    rows = []
    points: list[BetaPoint] = []

    for idx, cfg in enumerate(configs):
        p = cfg["p"]
        lam = cfg["lam"]
        delta = cfg["delta"]
        seed_cfg = seeds[idx]

        base = {
            "idx": idx, "p": p, "lam": lam, "delta": delta,
            "C": C, "X0": X0, "source": cfg["source"],
            "n_runs": n_runs, "max_steps": max_steps,
            "seed_cfg": seed_cfg,
        }

        # run simulator with optional Numba fallback
        try:
            sim = simulate_uncapped_first_passage(
                X0=X0, C=C, p=p, lam=lam,
                n_runs=n_runs, max_steps=max_steps,
                seed=seed_cfg, use_numba=use_numba,
            )
        except Exception as e:
            if use_numba:
                warnings.warn(
                    f"Numba failed at idx={idx}: {e}; "
                    "falling back to Python backend."
                )
                try:
                    sim = simulate_uncapped_first_passage(
                        X0=X0, C=C, p=p, lam=lam,
                        n_runs=n_runs, max_steps=max_steps,
                        seed=seed_cfg, use_numba=False,
                    )
                except Exception as e2:
                    rows.append({
                        **base,
                        "n_hits": 0, "no_hit_rate": np.nan,
                        "mu_hat": np.nan, "eta_hat": np.nan,
                        "sigma_eff_implied": np.nan,
                        "beta_implied": np.nan,
                        "passes_min_hits": False,
                        "in_validity_window": False,
                        "exclusion_reason":
                            f"sim_error:{type(e2).__name__}",
                    })
                    continue
            else:
                rows.append({
                    **base,
                    "n_hits": 0, "no_hit_rate": np.nan,
                    "mu_hat": np.nan, "eta_hat": np.nan,
                    "sigma_eff_implied": np.nan,
                    "beta_implied": np.nan,
                    "passes_min_hits": False,
                    "in_validity_window": False,
                    "exclusion_reason":
                        f"sim_error:{type(e).__name__}",
                })
                continue

        if sim.n_hits == 0:
            rows.append({
                **base,
                "n_hits": 0, "no_hit_rate": sim.no_hit_rate,
                "mu_hat": np.nan, "eta_hat": np.nan,
                "sigma_eff_implied": np.nan, "beta_implied": np.nan,
                "passes_min_hits": False,
                "in_validity_window": False,
                "exclusion_reason": "zero_hits",
            })
            continue

        try:
            bp = beta_implied_from_taus(
                sim.observed_taus, n_runs=sim.n_runs,
                p=p, delta=delta, C=C, X0=X0,
            )
        except Exception as e:
            rows.append({
                **base,
                "n_hits": int(sim.n_hits),
                "no_hit_rate": float(sim.no_hit_rate),
                "mu_hat": np.nan, "eta_hat": np.nan,
                "sigma_eff_implied": np.nan, "beta_implied": np.nan,
                "passes_min_hits": False,
                "in_validity_window": False,
                "exclusion_reason":
                    f"mle_error:{type(e).__name__}",
            })
            continue

        passes_hits = bool(bp.n_hits >= min_hits)
        in_window = bool(p <= p_max_validity and delta <= delta_max_validity)

        if not passes_hits:
            reason = f"low_hits({bp.n_hits}<{min_hits})"
        elif bp.beta_implied <= 0.0:
            reason = "beta<=0"
        elif not in_window:
            reason = f"outside_validity_window(p<={p_max_validity},"\
                     f"delta<={delta_max_validity})"
        else:
            reason = ""

        rows.append({
            **base,
            "n_hits": int(bp.n_hits),
            "no_hit_rate": float(bp.no_hit_rate),
            "mu_hat": float(bp.mu_hat),
            "eta_hat": float(bp.eta_hat),
            "sigma_eff_implied": float(bp.sigma_eff_implied),
            "beta_implied": float(bp.beta_implied),
            "passes_min_hits": passes_hits,
            "in_validity_window": in_window,
            "exclusion_reason": reason,
        })

        if passes_hits and bp.beta_implied > 0.0:
            points.append(bp)

    df_all = pd.DataFrame(rows)

    # dedupe by (p, lam) rounded to 4 decimals, keep first
    if not df_all.empty:
        df_all["_p_round"] = df_all["p"].round(4)
        df_all["_lam_round"] = df_all["lam"].round(4)
        df_all = df_all.drop_duplicates(
            subset=["_p_round", "_lam_round"], keep="first"
        ).drop(columns=["_p_round", "_lam_round"]).reset_index(drop=True)

    # validity-window filter
    df_validity = df_all[
        df_all["passes_min_hits"]
        & df_all["in_validity_window"]
        & (df_all["beta_implied"] > 0.0)
    ].copy().reset_index(drop=True)

    # build BetaPoint list aligned to df_validity (rebuild from rows)
    points_validity: list[BetaPoint] = []
    for _, row in df_validity.iterrows():
        points_validity.append(BetaPoint(
            p=float(row["p"]), delta=float(row["delta"]),
            lam=float(row["lam"]), C=int(row["C"]), X0=int(row["X0"]),
            n_runs=int(row["n_runs"]), n_hits=int(row["n_hits"]),
            no_hit_rate=float(row["no_hit_rate"]),
            mu_hat=float(row["mu_hat"]),
            eta_hat=float(row["eta_hat"]),
            sigma_eff_implied=float(row["sigma_eff_implied"]),
            beta_implied=float(row["beta_implied"]),
        ))

    # fit
    try:
        fit = fit_beta_nls_paper(points_validity, min_hits=min_hits)
        fit_dict = {
            "a": fit.a, "b": fit.b, "c": fit.c,
            "R2_raw_beta": fit.R2_raw_beta,
            "n_points_used": fit.n_points_used,
            "n_points_total": fit.n_points_total,
            "MAE_uncorrected_eta_pct": fit.MAE_uncorrected_eta_pct,
            "MAE_corrected_eta_pct": fit.MAE_corrected_eta_pct,
            "max_error_uncorrected_eta_pct":
                fit.max_error_uncorrected_eta_pct,
            "max_error_corrected_eta_pct":
                fit.max_error_corrected_eta_pct,
            "median_error_uncorrected_eta_pct":
                fit.median_error_uncorrected_eta_pct,
            "median_error_corrected_eta_pct":
                fit.median_error_corrected_eta_pct,
            "improvement_factor": fit.improvement_factor,
            "err_eta_pct_by_p": {
                f"({lo},{hi}]": v
                for (lo, hi), v in fit.err_eta_pct_by_p.items()
            },
            "err_eta_pct_by_delta": {
                f"({lo},{hi}]": v
                for (lo, hi), v in fit.err_eta_pct_by_delta.items()
            },
            "cov_diag": fit.cov_diag,
        }
    except Exception as e:
        fit_dict = {
            "error": f"{type(e).__name__}:{e}",
            "n_points_total": len(points_validity),
        }

    return df_all, df_validity, fit_dict
