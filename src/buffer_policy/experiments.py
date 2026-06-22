"""
Phase 4 experiments (modular).

This module is an orchestrator only. It does not introduce new mathematics.

Capped deterministic components: kernel, survival, E[tau], RTF, threshold,
hazard diagnostics.

Uncapped Monte Carlo component: beta calibration only.

Key formulas:
    g_RTF = K_f / E[tau]
    g_AB(T) = [K_f(1-S(T)) + K_p S(T)] / sum_{t=0}^{T-1} S(t)
    g_AB_best = min( min_T g_AB(T), g_RTF )
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from buffer_policy.age_based import age_based_curve
from buffer_policy.hazard import hazard_diagnostics
from buffer_policy.ig_beta import (
    BetaPoint,
    beta_implied_from_taus,
    fit_beta_log_model,
)
from buffer_policy.kernel import build_kernel
from buffer_policy.mdp import optimize_threshold
from buffer_policy.params import CostParams, ModelParams
from buffer_policy.simulation import simulate_uncapped_first_passage
from buffer_policy.survival import (
    expected_tau_linear_system,
    survival_from_kernel,
)
from buffer_policy.validation import audit_lemma1


def get_grids(mode: str) -> dict:
    """Return experiment grids for fast/full mode."""
    if mode == "fast":
        return {
            "cost_ratio_kp": [2, 5, 8, 10, 12, 15, 20],
            "heatmap_p": [0.05, 0.15, 0.25, 0.35],
            "heatmap_delta": [1.0, 5.0, 10.0, 15.0],
            "smax_values": [80, 120, 160, 180, 220, 260],
            "beta_p": [0.05, 0.15, 0.25, 0.35],
            "beta_delta": [1.0, 5.0, 15.0],
            "hazard_configs": [
                {"name": "baseline", "p": 0.15, "delta": 5.0,
                 "C": 100, "S_max": 180},
                {"name": "near_critical_d1", "p": 0.15, "delta": 1.0,
                 "C": 100, "S_max": 180},
                {"name": "high_p", "p": 0.30, "delta": 5.0,
                 "C": 100, "S_max": 180},
                {"name": "low_p", "p": 0.08, "delta": 5.0,
                 "C": 100, "S_max": 180},
            ],
        }

    if mode == "full":
        return {
            "cost_ratio_kp": [1, 2, 3, 5, 7, 8, 10, 12, 15, 18, 20,
                              25, 30, 40, 50],
            "heatmap_p": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            "heatmap_delta": [1.0, 3.0, 5.0, 8.0, 12.0, 15.0],
            "smax_values": [60, 80, 100, 120, 140, 160, 180,
                            200, 220, 260, 300],
            "beta_p": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
            "beta_delta": [1.0, 3.0, 5.0, 8.0, 12.0, 15.0],
            "hazard_configs": [
                {"name": "baseline", "p": 0.15, "delta": 5.0,
                 "C": 100, "S_max": 180},
                {"name": "near_critical_d1", "p": 0.15, "delta": 1.0,
                 "C": 100, "S_max": 180},
                {"name": "high_p", "p": 0.30, "delta": 5.0,
                 "C": 100, "S_max": 180},
                {"name": "low_p", "p": 0.08, "delta": 5.0,
                 "C": 100, "S_max": 180},
                {"name": "large_delta_12", "p": 0.15, "delta": 12.0,
                 "C": 100, "S_max": 180},
                {"name": "tight_smax_80", "p": 0.15, "delta": 5.0,
                 "C": 100, "S_max": 80},
            ],
        }

    raise ValueError(f"unknown mode: {mode}")


def robustness_scenarios() -> list[dict]:
    """Named robustness scenarios."""
    return [
        {"name": "baseline", "p": 0.15, "delta": 5.0,
         "C": 100, "S_max": 180, "K_p": 10},
        {"name": "low_p", "p": 0.08, "delta": 5.0,
         "C": 100, "S_max": 180, "K_p": 10},
        {"name": "high_p", "p": 0.30, "delta": 5.0,
         "C": 100, "S_max": 180, "K_p": 10},
        {"name": "near_critical", "p": 0.15, "delta": 1.0,
         "C": 100, "S_max": 180, "K_p": 10},
        {"name": "larger_delta", "p": 0.15, "delta": 12.0,
         "C": 100, "S_max": 180, "K_p": 10},
        {"name": "cheap_pm", "p": 0.15, "delta": 5.0,
         "C": 100, "S_max": 180, "K_p": 5},
        {"name": "expensive_pm", "p": 0.15, "delta": 5.0,
         "C": 100, "S_max": 180, "K_p": 20},
        {"name": "larger_capacity", "p": 0.15, "delta": 5.0,
         "C": 120, "S_max": 220, "K_p": 10},
    ]


def validate_physical(
    p: float, lam: float, C: int, S_max: int, X0: int, s_reset: int
) -> str:
    """Return empty string if valid, otherwise reason."""
    delta = (1.0 - p) * C - lam
    if delta <= 0:
        return f"delta<=0 ({delta})"
    if lam <= 0:
        return f"lam<=0 ({lam})"
    if S_max < s_reset:
        return f"S_max<s_reset ({S_max}<{s_reset})"
    if S_max < X0:
        return f"S_max<X0 ({S_max}<{X0})"
    return ""


def baseline_model_phase4() -> ModelParams:
    return ModelParams(p=0.15, lam=80.0, C=100, S_max=180, X0=50, s_reset=50)


def run_cost_ratio_experiment(
    mode: str, K_f: float = 100.0, T_age_max: int = 5000
) -> pd.DataFrame:
    """Vary K_p/K_f for fixed baseline physical parameters."""
    grids = get_grids(mode)
    mp = baseline_model_phase4()
    P = build_kernel(mp)
    E_tau = expected_tau_linear_system(P, mp)
    S = survival_from_kernel(P, mp, T_max=T_age_max, tol=1e-12)
    T_grid = np.arange(1, T_age_max + 1)

    rows = []
    for K_p in grids["cost_ratio_kp"]:
        cp = CostParams(K_f=K_f, K_p=float(K_p))
        g_RTF = K_f / E_tau
        s_star, g_star, _ = optimize_threshold(mp, cp, P=P)

        ab = age_based_curve(S, T_grid, cp.K_f, cp.K_p)
        g_ab_finite = float(np.min(ab))
        g_ab_best = min(g_ab_finite, g_RTF)

        rows.append({
            "K_f": K_f,
            "K_p": float(K_p),
            "kp_over_kf": float(K_p) / K_f,
            "s_star": int(s_star),
            "g_opt": float(g_star),
            "g_RTF": float(g_RTF),
            "g_AB_best_finite": g_ab_finite,
            "g_AB_best": float(g_ab_best),
            "ratio_opt_to_RTF": float(g_star / g_RTF),
            "ratio_opt_to_AB": float(g_star / g_ab_best),
            "E_tau": float(E_tau),
        })

    return pd.DataFrame(rows)


def run_heatmap_experiment(
    mode: str,
    K_f: float = 100.0,
    K_p: float = 10.0,
    C: int = 100,
    S_max: int = 180,
    X0: int = 50,
    s_reset: int = 50,
) -> pd.DataFrame:
    """Grid over p and delta."""
    grids = get_grids(mode)
    cp = CostParams(K_f=K_f, K_p=K_p)
    rows = []

    for p in grids["heatmap_p"]:
        for delta in grids["heatmap_delta"]:
            lam = (1.0 - p) * C - delta
            reason = validate_physical(p, lam, C, S_max, X0, s_reset)

            if reason:
                rows.append({
                    "p": p, "delta": delta, "lam": lam, "C": C,
                    "S_max": S_max, "X0": X0, "s_reset": s_reset,
                    "s_star": -1, "g_star": np.nan, "g_RTF": np.nan,
                    "ratio_star_to_RTF": np.nan,
                    "valid": False, "skip_reason": reason,
                })
                continue

            mp = ModelParams(p=p, lam=lam, C=C, S_max=S_max,
                             X0=X0, s_reset=s_reset)
            P = build_kernel(mp)
            E_tau = expected_tau_linear_system(P, mp)
            g_RTF = K_f / E_tau
            s_star, g_star, _ = optimize_threshold(mp, cp, P=P)

            rows.append({
                "p": p, "delta": delta, "lam": lam, "C": C,
                "S_max": S_max, "X0": X0, "s_reset": s_reset,
                "s_star": int(s_star),
                "g_star": float(g_star),
                "g_RTF": float(g_RTF),
                "ratio_star_to_RTF": float(g_star / g_RTF),
                "valid": True,
                "skip_reason": "",
            })

    return pd.DataFrame(rows)


def run_smax_experiment(
    mode: str, K_f: float = 100.0, K_p: float = 10.0
) -> pd.DataFrame:
    """Vary S_max for fixed baseline physical parameters."""
    grids = get_grids(mode)
    p, lam, C, X0, s_reset = 0.15, 80.0, 100, 50, 50
    cp = CostParams(K_f=K_f, K_p=K_p)
    rows = []

    for S_max in grids["smax_values"]:
        reason = validate_physical(p, lam, C, S_max, X0, s_reset)

        if reason:
            rows.append({
                "S_max": S_max, "p": p, "lam": lam, "C": C,
                "X0": X0, "s_reset": s_reset,
                "s_star": -1, "g_star": np.nan, "g_RTF": np.nan,
                "E_tau": np.nan, "ratio_star_to_RTF": np.nan,
                "valid": False, "skip_reason": reason,
            })
            continue

        mp = ModelParams(p=p, lam=lam, C=C, S_max=S_max,
                         X0=X0, s_reset=s_reset)
        P = build_kernel(mp)
        E_tau = expected_tau_linear_system(P, mp)
        g_RTF = K_f / E_tau
        s_star, g_star, _ = optimize_threshold(mp, cp, P=P)

        rows.append({
            "S_max": S_max, "p": p, "lam": lam, "C": C,
            "X0": X0, "s_reset": s_reset,
            "s_star": int(s_star),
            "g_star": float(g_star),
            "g_RTF": float(g_RTF),
            "E_tau": float(E_tau),
            "ratio_star_to_RTF": float(g_star / g_RTF),
            "valid": True,
            "skip_reason": "",
        })

    return pd.DataFrame(rows)


def run_robustness_experiment(
    mode: str, K_f: float = 100.0, X0: int = 50, s_reset: int = 50
) -> pd.DataFrame:
    """Run named robustness scenarios and Lemma 1 audit."""
    rows = []

    for sc in robustness_scenarios():
        p = sc["p"]
        delta = sc["delta"]
        C = sc["C"]
        S_max = sc["S_max"]
        K_p = sc["K_p"]
        lam = (1.0 - p) * C - delta

        reason = validate_physical(p, lam, C, S_max, X0, s_reset)

        if reason:
            rows.append({
                "scenario": sc["name"], "p": p, "delta": delta, "lam": lam,
                "C": C, "S_max": S_max, "K_f": K_f, "K_p": K_p,
                "s_star": -1, "g_star": np.nan, "g_RTF": np.nan,
                "ratio_star_to_RTF": np.nan,
                "lemma1_premise": False,
                "lemma1_holds": False,
                "valid": False,
                "skip_reason": reason,
            })
            continue

        mp = ModelParams(p=p, lam=lam, C=C, S_max=S_max,
                         X0=X0, s_reset=s_reset)
        cp = CostParams(K_f=K_f, K_p=float(K_p))
        P = build_kernel(mp)
        E_tau = expected_tau_linear_system(P, mp)
        g_RTF = K_f / E_tau
        s_star, g_star, _ = optimize_threshold(mp, cp, P=P)
        lem = audit_lemma1(mp, cp, P=P)

        rows.append({
            "scenario": sc["name"],
            "p": p, "delta": delta, "lam": lam,
            "C": C, "S_max": S_max,
            "K_f": K_f, "K_p": float(K_p),
            "s_star": int(s_star),
            "g_star": float(g_star),
            "g_RTF": float(g_RTF),
            "ratio_star_to_RTF": float(g_star / g_RTF),
            "lemma1_premise": bool(lem.premise_holds),
            "lemma1_holds": bool(lem.holds),
            "valid": True,
            "skip_reason": "",
        })

    return pd.DataFrame(rows)


def run_hazard_diagnostics_experiment(
    mode: str,
    X0: int = 50,
    s_reset: int = 50,
    T_max: int = 50_000,
    tol: float = 1e-10,
    window: int = 25,
) -> pd.DataFrame:
    """Capped deterministic hazard diagnostics."""
    grids = get_grids(mode)
    rows = []

    for cfg in grids["hazard_configs"]:
        p = cfg["p"]
        delta = cfg["delta"]
        C = cfg["C"]
        S_max = cfg["S_max"]
        lam = (1.0 - p) * C - delta

        reason = validate_physical(p, lam, C, S_max, X0, s_reset)

        if reason:
            rows.append({
                "config": cfg["name"], "p": p, "delta": delta, "lam": lam,
                "C": C, "S_max": S_max,
                "DFR_strict_raw": False,
                "DFR_strict_smoothed": False,
                "DFR_approx_smoothed": False,
                "frac_decreasing_raw": np.nan,
                "frac_decreasing_smoothed": np.nan,
                "h_initial": np.nan,
                "h_final": np.nan,
                "n_hazard_points": 0,
                "n_hazard_smoothed": 0,
                "S_last": np.nan,
                "survival_reached_1e8_tol": False,
                "valid": False,
                "skip_reason": reason,
            })
            continue

        mp = ModelParams(p=p, lam=lam, C=C, S_max=S_max,
                         X0=X0, s_reset=s_reset)
        P = build_kernel(mp)
        S = survival_from_kernel(P, mp, T_max=T_max, tol=tol)
        diag = hazard_diagnostics(S, window=window, survival_tol=1e-8)

        rows.append({
            "config": cfg["name"],
            "p": p, "delta": delta, "lam": lam,
            "C": C, "S_max": S_max,
            "DFR_strict_raw": bool(diag.DFR_strict_raw),
            "DFR_strict_smoothed": bool(diag.DFR_strict_smoothed),
            "DFR_approx_smoothed": bool(diag.DFR_approx_smoothed),
            "frac_decreasing_raw": float(diag.frac_decreasing_raw),
            "frac_decreasing_smoothed": float(diag.frac_decreasing_smoothed),
            "h_initial": float(diag.h_initial),
            "h_final": float(diag.h_final),
            "n_hazard_points": int(diag.n_hazard_points),
            "n_hazard_smoothed": int(diag.n_hazard_smoothed),
            "S_last": float(S[-1]),
            "survival_reached_1e8_tol": bool(S[-1] <= 1e-8),
            "valid": True,
            "skip_reason": "",
        })

    return pd.DataFrame(rows)


def run_beta_calibration_experiment(
    mode: str,
    n_runs: int,
    max_steps: int,
    seed: int,
    C: int = 100,
    X0: int = 50,
    min_hits: int = 500,
) -> tuple[pd.DataFrame, dict | None]:
    """Optional uncapped beta calibration."""
    grids = get_grids(mode)

    if mode == "fast":
        eff_n_runs = min(n_runs, 2000)
        eff_max_steps = min(max_steps, 20_000)
    else:
        eff_n_runs = n_runs
        eff_max_steps = max_steps

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(len(grids["beta_p"]) * len(grids["beta_delta"]))

    rows = []
    points: list[BetaPoint] = []
    k = 0

    for p in grids["beta_p"]:
        for delta in grids["beta_delta"]:
            lam = (1.0 - p) * C - delta
            child_seed = int(children[k].generate_state(1, dtype=np.uint32)[0])
            k += 1

            base = {
                "p": p, "delta": delta, "lam": lam, "C": C, "X0": X0,
                "n_runs": eff_n_runs, "max_steps": eff_max_steps,
                "seed": child_seed,
            }

            if lam <= 0:
                rows.append({
                    **base,
                    "n_hits": 0, "no_hit_rate": np.nan,
                    "mu_hat": np.nan, "eta_hat": np.nan,
                    "sigma_eff_implied": np.nan, "beta_implied": np.nan,
                    "included_in_fit": False,
                    "exclusion_reason": "lam<=0",
                })
                continue

            try:
                sim = simulate_uncapped_first_passage(
                    X0=X0, C=C, p=p, lam=lam,
                    n_runs=eff_n_runs, max_steps=eff_max_steps,
                    seed=child_seed, use_numba=True,
                )

                if sim.n_hits == 0:
                    rows.append({
                        **base,
                        "n_hits": 0, "no_hit_rate": sim.no_hit_rate,
                        "mu_hat": np.nan, "eta_hat": np.nan,
                        "sigma_eff_implied": np.nan, "beta_implied": np.nan,
                        "included_in_fit": False,
                        "exclusion_reason": "zero_hits",
                    })
                    continue

                bp = beta_implied_from_taus(
                    sim.observed_taus, n_runs=sim.n_runs,
                    p=p, delta=delta, C=C, X0=X0,
                )
                points.append(bp)

                included = bool(bp.n_hits >= min_hits and bp.beta_implied > 0.0)
                if bp.n_hits < min_hits:
                    reason = f"low_hits({bp.n_hits}<{min_hits})"
                elif bp.beta_implied <= 0.0:
                    reason = "beta<=0"
                else:
                    reason = ""

                rows.append({
                    **base,
                    "n_hits": bp.n_hits,
                    "no_hit_rate": bp.no_hit_rate,
                    "mu_hat": bp.mu_hat,
                    "eta_hat": bp.eta_hat,
                    "sigma_eff_implied": bp.sigma_eff_implied,
                    "beta_implied": bp.beta_implied,
                    "included_in_fit": included,
                    "exclusion_reason": reason,
                })

            except Exception as exc:
                rows.append({
                    **base,
                    "n_hits": 0, "no_hit_rate": np.nan,
                    "mu_hat": np.nan, "eta_hat": np.nan,
                    "sigma_eff_implied": np.nan, "beta_implied": np.nan,
                    "included_in_fit": False,
                    "exclusion_reason": f"{type(exc).__name__}:{exc}",
                })

    df = pd.DataFrame(rows)

    try:
        fit = fit_beta_log_model(points, min_hits=min_hits)
        fit_dict = asdict(fit)
    except Exception as exc:
        fit_dict = {
            "error": f"{type(exc).__name__}:{exc}",
            "n_points_total": len(points),
            "n_points_included_csv": int(
                df["included_in_fit"].astype(bool).sum()
            ) if len(df) else 0,
        }

    return df, fit_dict
