"""
Paper-grade experiments. Reproduces Tables 3-8 of the paper exactly.

Differences from `experiments.py`:
  - Hardcoded paper grids (PAPER_P_GRID, PAPER_UTIL_GRID, etc.).
  - Hazard diagnostics use stricter tolerances (monotone_tol=1e-10,
    strict_tol=1e-8) implemented locally as `hazard_diagnostics_paper`.
  - `survival_fixed_horizon` forces a fixed-length survival vector to
    preserve T*_finite values (1683, 4583, etc.) for age-based curves.
  - `T_star_finite` is always reported (never replaced by inf).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from buffer_policy.age_based import age_based_curve
from buffer_policy.kernel import build_kernel
from buffer_policy.mdp import optimize_threshold
from buffer_policy.params import CostParams, ModelParams
from buffer_policy.survival import (
    expected_tau_linear_system,
    survival_from_kernel,
)
from buffer_policy.validation import audit_lemma1


# ----------------------------------------------------- paper grids
PAPER_P_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
PAPER_UTIL_GRID = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
PAPER_COST_RATIOS = [5, 10, 20, 50, 100]   # K_f/K_p with K_p=10 fixed
PAPER_SMAX_VALUES = [100, 140, 180, 220, 260, 300]
PAPER_SMAX_CONFIGS = [
    {"name": "p015_u080", "p": 0.15, "u": 0.80, "C": 100},
    {"name": "p010_u085", "p": 0.10, "u": 0.85, "C": 100},
    {"name": "p005_u080", "p": 0.05, "u": 0.80, "C": 100},
]
PAPER_MISSPEC_ERR_PCT = [-30, -20, -10, 0, 10, 20, 30]


# ----------------------------------------------------- helpers
def survival_fixed_horizon(P, mp, T_max: int) -> np.ndarray:
    """
    Force a survival vector of length exactly T_max+1.
    Uses tol=-1.0 to disable early truncation.
    """
    return survival_from_kernel(P, mp, T_max=T_max, tol=-1.0)


def _validate_physical_paper(
    p: float, lam: float, C: int, S_max: int, X0: int, s_reset: int
) -> str:
    delta = (1.0 - p) * C - lam
    if delta <= 0:
        return f"delta<=0 ({delta:.4f})"
    if lam <= 0:
        return f"lam<=0 ({lam:.4f})"
    if S_max < s_reset:
        return f"S_max<s_reset ({S_max}<{s_reset})"
    if S_max < X0:
        return f"S_max<X0 ({S_max}<{X0})"
    return ""


def _baseline_paper_mp() -> ModelParams:
    return ModelParams(p=0.15, lam=80.0, C=100, S_max=180, X0=50, s_reset=50)


def _baseline_paper_cp() -> CostParams:
    return CostParams(K_f=100.0, K_p=10.0)


# ----------------------------------------------------- hazard diagnostics (paper)
@dataclass(frozen=True)
class HazardDiagnosticsPaper:
    h_initial: float
    h_final: float
    h_min: float
    h_max: float
    DFR_strict_raw: bool
    DFR_strict_smoothed: bool
    DFR_approx_smoothed: bool
    frac_decreasing_raw: float
    frac_decreasing_smoothed: float
    n_increases_smoothed: int
    n_hazard_points: int
    n_hazard_smoothed: int
    smoothing_window: int
    survival_floor: float


def hazard_diagnostics_paper(
    S: np.ndarray,
    smooth_window: int = 25,
    survival_floor: float = 1e-8,
) -> HazardDiagnosticsPaper:
    """
    Paper-grade DFR diagnostic.

    - frac_decreasing := mean(diff_smoothed <= 1e-10)
    - n_increases     := sum(diff_smoothed > 1e-8)
    - DFR_strict_smoothed := (n_increases == 0)
    - DFR_approx_smoothed := (frac_decreasing >= 0.95)
    """
    S = np.asarray(S, dtype=np.float64)
    if S.size < 2:
        return HazardDiagnosticsPaper(
            h_initial=float("nan"), h_final=float("nan"),
            h_min=float("nan"), h_max=float("nan"),
            DFR_strict_raw=False, DFR_strict_smoothed=False,
            DFR_approx_smoothed=False,
            frac_decreasing_raw=float("nan"),
            frac_decreasing_smoothed=float("nan"),
            n_increases_smoothed=0,
            n_hazard_points=0, n_hazard_smoothed=0,
            smoothing_window=int(smooth_window),
            survival_floor=float(survival_floor),
        )

    denom = S[:-1]
    h_raw_full = (S[:-1] - S[1:]) / np.where(denom > 0.0, denom, 1.0)
    mask = denom > survival_floor
    h_raw = h_raw_full[mask]
    n_raw = h_raw.size

    if n_raw < 2:
        h_init = float(h_raw[0]) if n_raw == 1 else float("nan")
        h_fin = float(h_raw[-1]) if n_raw == 1 else float("nan")
        return HazardDiagnosticsPaper(
            h_initial=h_init, h_final=h_fin,
            h_min=h_init if n_raw == 1 else float("nan"),
            h_max=h_init if n_raw == 1 else float("nan"),
            DFR_strict_raw=False, DFR_strict_smoothed=False,
            DFR_approx_smoothed=False,
            frac_decreasing_raw=float("nan"),
            frac_decreasing_smoothed=float("nan"),
            n_increases_smoothed=0,
            n_hazard_points=int(n_raw), n_hazard_smoothed=0,
            smoothing_window=int(smooth_window),
            survival_floor=float(survival_floor),
        )

    diff_raw = np.diff(h_raw)
    dfr_strict_raw = bool(np.all(diff_raw <= 1e-12))
    frac_dec_raw = float((diff_raw <= 1e-10).mean())

    if n_raw >= smooth_window:
        kernel = np.ones(smooth_window, dtype=np.float64) / smooth_window
        h_smooth = np.convolve(h_raw, kernel, mode="valid")
    else:
        h_smooth = h_raw.copy()

    if h_smooth.size < 2:
        return HazardDiagnosticsPaper(
            h_initial=float(h_raw[0]),
            h_final=float(h_raw[-1]),
            h_min=float(h_raw.min()),
            h_max=float(h_raw.max()),
            DFR_strict_raw=dfr_strict_raw,
            DFR_strict_smoothed=False,
            DFR_approx_smoothed=False,
            frac_decreasing_raw=frac_dec_raw,
            frac_decreasing_smoothed=float("nan"),
            n_increases_smoothed=0,
            n_hazard_points=int(n_raw),
            n_hazard_smoothed=int(h_smooth.size),
            smoothing_window=int(smooth_window),
            survival_floor=float(survival_floor),
        )

    diff_smooth = np.diff(h_smooth)
    n_increases = int((diff_smooth > 1e-8).sum())
    frac_dec_smooth = float((diff_smooth <= 1e-10).mean())
    dfr_strict_smooth = bool(n_increases == 0)
    dfr_approx_smooth = bool(frac_dec_smooth >= 0.95)

    return HazardDiagnosticsPaper(
        h_initial=float(h_raw[0]),
        h_final=float(h_raw[-1]),
        h_min=float(h_raw.min()),
        h_max=float(h_raw.max()),
        DFR_strict_raw=dfr_strict_raw,
        DFR_strict_smoothed=dfr_strict_smooth,
        DFR_approx_smoothed=dfr_approx_smooth,
        frac_decreasing_raw=frac_dec_raw,
        frac_decreasing_smoothed=frac_dec_smooth,
        n_increases_smoothed=n_increases,
        n_hazard_points=int(n_raw),
        n_hazard_smoothed=int(h_smooth.size),
        smoothing_window=int(smooth_window),
        survival_floor=float(survival_floor),
    )


# ----------------------------------------------------- baseline
def run_baseline_paper() -> dict:
    """Baseline single-config paper result."""
    mp = _baseline_paper_mp()
    cp = _baseline_paper_cp()
    P = build_kernel(mp)
    E_tau = expected_tau_linear_system(P, mp)
    g_RTF = cp.K_f / E_tau
    s_star, g_star, _ = optimize_threshold(mp, cp, P=P)
    return {
        "p": mp.p, "lam": mp.lam, "C": mp.C, "S_max": mp.S_max,
        "X0": mp.X0, "s_reset": mp.s_reset,
        "K_f": cp.K_f, "K_p": cp.K_p,
        "E_tau": float(E_tau),
        "g_RTF": float(g_RTF),
        "s_star": int(s_star),
        "g_star": float(g_star),
    }


# ----------------------------------------------------- Table 4: cost ratio
def run_cost_ratio_paper(
    K_p_fixed: float = 10.0, T_age_max: int = 20_000
) -> pd.DataFrame:
    """Table 4: vary K_f/K_p with K_p=10 fixed."""
    mp = _baseline_paper_mp()
    P = build_kernel(mp)
    E_tau = expected_tau_linear_system(P, mp)
    S = survival_fixed_horizon(P, mp, T_max=T_age_max)
    T_grid = np.arange(1, T_age_max + 1)

    rows = []
    for ratio in PAPER_COST_RATIOS:
        K_f = float(ratio) * K_p_fixed
        cp = CostParams(K_f=K_f, K_p=K_p_fixed)
        g_RTF = K_f / E_tau
        s_star, g_star, _ = optimize_threshold(mp, cp, P=P)

        ab = age_based_curve(S, T_grid, cp.K_f, cp.K_p)
        idx_min = int(np.argmin(ab))
        T_star_finite = int(T_grid[idx_min])
        g_ab_finite = float(ab[idx_min])
        g_ab_best = min(g_ab_finite, g_RTF)

        gain_vs_RTF_pct = 100.0 * (g_RTF - g_star) / g_RTF
        gain_vs_AB_pct = 100.0 * (g_ab_best - g_star) / g_ab_best

        rows.append({
            "K_f_over_K_p": int(ratio),
            "K_f": K_f, "K_p": K_p_fixed,
            "s_star": int(s_star),
            "g_CBM": float(g_star),
            "g_RTF": float(g_RTF),
            "g_AB_finite": g_ab_finite,
            "g_AB_best": float(g_ab_best),
            "T_star_finite": T_star_finite,
            "gain_vs_RTF_pct": float(gain_vs_RTF_pct),
            "gain_vs_AB_best_pct": float(gain_vs_AB_pct),
            "E_tau": float(E_tau),
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------- Tables 5 & 8: heatmap
def run_heatmap_paper(
    K_f: float = 100.0, K_p: float = 10.0,
    C: int = 100, S_max: int = 180,
    X0: int = 50, s_reset: int = 50,
    T_age_max: int = 20_000,
) -> pd.DataFrame:
    """Tables 5 & 8: 6x6 grid of (p, util) -> 36 rows, 21 valid (delta>0)."""
    cp = CostParams(K_f=K_f, K_p=K_p)
    rows = []

    for p in PAPER_P_GRID:
        for u in PAPER_UTIL_GRID:
            lam = u * C
            delta = (1.0 - p) * C - lam
            reason = ""
            if delta <= 0:
                reason = "delta<=0"
            else:
                reason = _validate_physical_paper(
                    p, lam, C, S_max, X0, s_reset
                )

            if reason:
                rows.append({
                    "p": p, "u": u, "lam": lam, "delta": delta, "C": C,
                    "S_max": S_max, "X0": X0, "s_reset": s_reset,
                    "s_star": -1,
                    "g_CBM": np.nan, "g_RTF": np.nan,
                    "g_AB_best": np.nan, "T_star_finite": -1,
                    "ratio_CBM_to_RTF": np.nan,
                    "valid": False, "skip_reason": reason,
                })
                continue

            mp = ModelParams(p=p, lam=lam, C=C, S_max=S_max,
                             X0=X0, s_reset=s_reset)
            P = build_kernel(mp)
            E_tau = expected_tau_linear_system(P, mp)
            g_RTF = K_f / E_tau
            s_star, g_star, _ = optimize_threshold(mp, cp, P=P)

            S = survival_fixed_horizon(P, mp, T_max=T_age_max)
            T_grid = np.arange(1, T_age_max + 1)
            ab = age_based_curve(S, T_grid, cp.K_f, cp.K_p)
            idx_min = int(np.argmin(ab))
            T_star_finite = int(T_grid[idx_min])
            g_ab_finite = float(ab[idx_min])
            g_ab_best = min(g_ab_finite, g_RTF)

            rows.append({
                "p": p, "u": u, "lam": lam, "delta": delta, "C": C,
                "S_max": S_max, "X0": X0, "s_reset": s_reset,
                "s_star": int(s_star),
                "g_CBM": float(g_star),
                "g_RTF": float(g_RTF),
                "g_AB_best": float(g_ab_best),
                "T_star_finite": T_star_finite,
                "ratio_CBM_to_RTF": float(g_star / g_RTF),
                "E_tau": float(E_tau),
                "valid": True,
                "skip_reason": "",
            })

    df = pd.DataFrame(rows)
    n_valid = int(df["valid"].sum())
    assert n_valid == 21, f"expected 21 valid heatmap rows, got {n_valid}"
    return df


# ----------------------------------------------------- Table 6: S_max sweep
def run_smax_paper(
    K_f: float = 100.0, K_p: float = 10.0,
    X0: int = 50, s_reset: int = 50,
) -> pd.DataFrame:
    """Table 6: 3 configs x 6 S_max = 18 rows."""
    cp = CostParams(K_f=K_f, K_p=K_p)
    rows = []

    for cfg in PAPER_SMAX_CONFIGS:
        p = cfg["p"]
        u = cfg["u"]
        C = cfg["C"]
        lam = u * C

        for S_max in PAPER_SMAX_VALUES:
            reason = _validate_physical_paper(
                p, lam, C, S_max, X0, s_reset
            )
            if reason:
                rows.append({
                    "config": cfg["name"], "p": p, "u": u, "lam": lam, "C": C,
                    "S_max": S_max, "X0": X0, "s_reset": s_reset,
                    "E_tau": np.nan, "g_RTF": np.nan,
                    "s_star": -1, "g_CBM": np.nan,
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
                "config": cfg["name"], "p": p, "u": u, "lam": lam, "C": C,
                "S_max": S_max, "X0": X0, "s_reset": s_reset,
                "E_tau": float(E_tau),
                "g_RTF": float(g_RTF),
                "s_star": int(s_star),
                "g_CBM": float(g_star),
                "valid": True,
                "skip_reason": "",
            })

    df = pd.DataFrame(rows)
    assert len(df) == 18, f"expected 18 smax rows, got {len(df)}"
    return df


def smax_summary(df_smax: pd.DataFrame) -> pd.DataFrame:
    """Summary stats per S_max config."""
    rows = []
    for cfg_name, sub in df_smax[df_smax["valid"]].groupby("config"):
        rows.append({
            "config": cfg_name,
            "s_star_min": int(sub["s_star"].min()),
            "s_star_max": int(sub["s_star"].max()),
            "s_star_range": int(sub["s_star"].max() - sub["s_star"].min()),
            "E_tau_min": float(sub["E_tau"].min()),
            "E_tau_max": float(sub["E_tau"].max()),
            "E_tau_ratio": float(sub["E_tau"].max() / sub["E_tau"].min()),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------- Table 7: misspecification
def run_misspecification_paper(
    K_f: float = 100.0, K_p: float = 10.0,
) -> pd.DataFrame:
    """Table 7: regret under +/-30% misspecification of p and lambda."""
    mp_true = _baseline_paper_mp()
    cp = CostParams(K_f=K_f, K_p=K_p)
    P_true = build_kernel(mp_true)
    s_star_true, g_true, _ = optimize_threshold(mp_true, cp, P=P_true)

    rows = []

    # ---- p errors
    for err in PAPER_MISSPEC_ERR_PCT:
        p_est = mp_true.p * (1.0 + err / 100.0)
        if not (0.0 < p_est < 1.0):
            rows.append({
                "param_err": "p", "err_pct": err,
                "p_est": p_est, "lam_est": mp_true.lam,
                "s_star_est": -1, "s_star_true": int(s_star_true),
                "g_mis": np.nan, "g_true": float(g_true),
                "regret_pct": np.nan,
                "valid": False, "skip_reason": "p_est out of (0,1)",
            })
            continue

        delta_est = (1.0 - p_est) * mp_true.C - mp_true.lam
        if delta_est <= 0:
            rows.append({
                "param_err": "p", "err_pct": err,
                "p_est": p_est, "lam_est": mp_true.lam,
                "s_star_est": -1, "s_star_true": int(s_star_true),
                "g_mis": np.nan, "g_true": float(g_true),
                "regret_pct": np.nan,
                "valid": False, "skip_reason": "delta_est<=0",
            })
            continue

        mp_est = ModelParams(p=p_est, lam=mp_true.lam, C=mp_true.C,
                             S_max=mp_true.S_max,
                             X0=mp_true.X0, s_reset=mp_true.s_reset)
        P_est = build_kernel(mp_est)
        s_star_est, _, _ = optimize_threshold(mp_est, cp, P=P_est)

        # Evaluate the misspecified policy under TRUE dynamics
        from buffer_policy.mdp import evaluate_threshold_policy
        g_mis = evaluate_threshold_policy(
            int(s_star_est), mp_true, cp, P=P_true
        )
        regret_pct = 100.0 * (g_mis - g_true) / g_true

        rows.append({
            "param_err": "p", "err_pct": err,
            "p_est": p_est, "lam_est": mp_true.lam,
            "s_star_est": int(s_star_est),
            "s_star_true": int(s_star_true),
            "g_mis": float(g_mis), "g_true": float(g_true),
            "regret_pct": float(regret_pct),
            "valid": True, "skip_reason": "",
        })

    # ---- lambda errors
    for err in PAPER_MISSPEC_ERR_PCT:
        lam_est = mp_true.lam * (1.0 + err / 100.0)
        delta_est = (1.0 - mp_true.p) * mp_true.C - lam_est

        if lam_est <= 0 or delta_est <= 0:
            rows.append({
                "param_err": "lam", "err_pct": err,
                "p_est": mp_true.p, "lam_est": lam_est,
                "s_star_est": -1, "s_star_true": int(s_star_true),
                "g_mis": np.nan, "g_true": float(g_true),
                "regret_pct": np.nan,
                "valid": False,
                "skip_reason": "lam_est<=0 or delta_est<=0",
            })
            continue

        mp_est = ModelParams(p=mp_true.p, lam=lam_est, C=mp_true.C,
                             S_max=mp_true.S_max,
                             X0=mp_true.X0, s_reset=mp_true.s_reset)
        P_est = build_kernel(mp_est)
        s_star_est, _, _ = optimize_threshold(mp_est, cp, P=P_est)

        from buffer_policy.mdp import evaluate_threshold_policy
        g_mis = evaluate_threshold_policy(
            int(s_star_est), mp_true, cp, P=P_true
        )
        regret_pct = 100.0 * (g_mis - g_true) / g_true

        rows.append({
            "param_err": "lam", "err_pct": err,
            "p_est": mp_true.p, "lam_est": lam_est,
            "s_star_est": int(s_star_est),
            "s_star_true": int(s_star_true),
            "g_mis": float(g_mis), "g_true": float(g_true),
            "regret_pct": float(regret_pct),
            "valid": True, "skip_reason": "",
        })

    return pd.DataFrame(rows)


# ----------------------------------------------------- Table 3: hazard
def run_hazard_paper(
    X0: int = 50, s_reset: int = 50, S_max: int = 180,
    T_max: int = 50_000, smooth_window: int = 25,
) -> pd.DataFrame:
    """Table 3: hazard DFR diagnostics over PAPER_P_GRID x PAPER_UTIL_GRID."""
    rows = []
    for p in PAPER_P_GRID:
        for u in PAPER_UTIL_GRID:
            C = 100
            lam = u * C
            delta = (1.0 - p) * C - lam

            reason = ""
            if delta <= 0:
                reason = "delta<=0"
            else:
                reason = _validate_physical_paper(
                    p, lam, C, S_max, X0, s_reset
                )

            if reason:
                rows.append({
                    "p": p, "u": u, "lam": lam, "delta": delta, "C": C,
                    "S_max": S_max,
                    "h_initial": np.nan, "h_final": np.nan,
                    "h_min": np.nan, "h_max": np.nan,
                    "DFR_strict_raw": False,
                    "DFR_strict_smoothed": False,
                    "DFR_approx_smoothed": False,
                    "frac_decreasing_raw": np.nan,
                    "frac_decreasing_smoothed": np.nan,
                    "n_increases_smoothed": 0,
                    "n_hazard_points": 0,
                    "valid": False, "skip_reason": reason,
                })
                continue

            mp = ModelParams(p=p, lam=lam, C=C, S_max=S_max,
                             X0=X0, s_reset=s_reset)
            P = build_kernel(mp)
            S = survival_from_kernel(P, mp, T_max=T_max, tol=1e-12)
            diag = hazard_diagnostics_paper(
                S, smooth_window=smooth_window, survival_floor=1e-8,
            )

            rows.append({
                "p": p, "u": u, "lam": lam, "delta": delta, "C": C,
                "S_max": S_max,
                "h_initial": float(diag.h_initial),
                "h_final": float(diag.h_final),
                "h_min": float(diag.h_min),
                "h_max": float(diag.h_max),
                "DFR_strict_raw": bool(diag.DFR_strict_raw),
                "DFR_strict_smoothed": bool(diag.DFR_strict_smoothed),
                "DFR_approx_smoothed": bool(diag.DFR_approx_smoothed),
                "frac_decreasing_raw": float(diag.frac_decreasing_raw),
                "frac_decreasing_smoothed":
                    float(diag.frac_decreasing_smoothed),
                "n_increases_smoothed": int(diag.n_increases_smoothed),
                "n_hazard_points": int(diag.n_hazard_points),
                "valid": True, "skip_reason": "",
            })

    df = pd.DataFrame(rows)
    n_valid = int(df["valid"].sum())
    assert n_valid == 21, f"expected 21 valid hazard rows, got {n_valid}"
    return df


def hazard_summary(df_hazard: pd.DataFrame) -> dict:
    """Aggregate hazard summary stats."""
    valid = df_hazard[df_hazard["valid"]]
    n_valid = int(len(valid))
    if n_valid == 0:
        return {
            "total_configs": 0,
            "DFR_strict_total": 0,
            "DFR_approximate_total": 0,
            "frac_strict": float("nan"),
            "frac_approx": float("nan"),
        }
    n_strict = int(valid["DFR_strict_smoothed"].sum())
    n_approx = int(valid["DFR_approx_smoothed"].sum())
    return {
        "total_configs": n_valid,
        "DFR_strict_total": n_strict,
        "DFR_approximate_total": n_approx,
        "frac_strict": float(n_strict / n_valid),
        "frac_approx": float(n_approx / n_valid),
    }


# ----------------------------------------------------- Lemma 1 audit
def run_lemma1_audit_paper(
    K_f: float = 100.0, K_p: float = 10.0,
    C: int = 100, S_max: int = 180,
    X0: int = 50, s_reset: int = 50,
) -> pd.DataFrame:
    """Lemma 1 audit over the 6x6 paper grid. 21 valid configs."""
    cp = CostParams(K_f=K_f, K_p=K_p)
    rows = []

    for p in PAPER_P_GRID:
        for u in PAPER_UTIL_GRID:
            lam = u * C
            delta = (1.0 - p) * C - lam
            reason = ""
            if delta <= 0:
                reason = "delta<=0"
            else:
                reason = _validate_physical_paper(
                    p, lam, C, S_max, X0, s_reset
                )

            if reason:
                rows.append({
                    "p": p, "u": u, "lam": lam, "delta": delta,
                    "lemma1_premise": False,
                    "lemma1_holds": False,
                    "valid": False, "skip_reason": reason,
                })
                continue

            mp = ModelParams(p=p, lam=lam, C=C, S_max=S_max,
                             X0=X0, s_reset=s_reset)
            P = build_kernel(mp)
            lem = audit_lemma1(mp, cp, P=P)

            rows.append({
                "p": p, "u": u, "lam": lam, "delta": delta,
                "lemma1_premise": bool(lem.premise_holds),
                "lemma1_holds": bool(lem.holds),
                "valid": True, "skip_reason": "",
            })

    df = pd.DataFrame(rows)
    n_valid = int(df["valid"].sum())
    assert n_valid == 21, f"expected 21 valid lemma1 rows, got {n_valid}"
    return df
