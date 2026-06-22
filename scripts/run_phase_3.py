"""
Phase 3 validation script.

Checks:
  1. Simulator reproducibility (Numba backend).
  2. Simulator reproducibility (Python backend).
  3. IG MLE on synthetic IG data recovers (mu, eta) within 5%.
  4. Log-linear beta fit on synthetic exact data within 10%, R^2 > 0.99.
  5. Hazard diagnostics on a mixture of exponentials report DFR_strict.
  6. Persist phase 3 artefact JSON.
  7. Run pytest.

Usage: python scripts/run_phase_3.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from buffer_policy.hazard import hazard_diagnostics  # noqa: E402
from buffer_policy.ig_beta import (  # noqa: E402
    BetaPoint,
    fit_beta_log_model,
    ig_mle_closed_form,
    sample_inverse_gaussian,
)
from buffer_policy.io_utils import (  # noqa: E402
    ensure_dir,
    make_metadata,
    save_json,
)
from buffer_policy.simulation import (  # noqa: E402
    simulate_uncapped_first_passage,
)


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def _check_simulator_backend(seed_master: int, use_numba: bool) -> dict:
    kw = dict(
        X0=20, C=40, p=0.15, lam=30.0,
        n_runs=200, max_steps=5000,
        use_numba=use_numba,
    )
    r1 = simulate_uncapped_first_passage(**kw, seed=seed_master)
    r2 = simulate_uncapped_first_passage(**kw, seed=seed_master)
    r3 = simulate_uncapped_first_passage(**kw, seed=seed_master + 1)
    same = (
        np.array_equal(r1.taus, r2.taus)
        and np.array_equal(r1.hit, r2.hit)
    )
    diff = (
        not np.array_equal(r1.taus, r3.taus)
        or not np.array_equal(r1.hit, r3.hit)
    )
    return {
        "backend": r1.backend,
        "same_seed_identical": bool(same),
        "diff_seed_different": bool(diff),
        "n_runs": r1.n_runs,
        "n_hits": r1.n_hits,
        "no_hit_rate": r1.no_hit_rate,
    }


def main() -> int:
    banner("Phase 3 validation")
    seed_master = 20260429

    # ---- 1. Simulator reproducibility (Numba)
    banner("1. Simulator reproducibility (Numba backend)")
    info_numba = _check_simulator_backend(seed_master, use_numba=True)
    print(f"  backend             = {info_numba['backend']}")
    print(f"  same seed identical = {info_numba['same_seed_identical']}")
    print(f"  diff seed different = {info_numba['diff_seed_different']}")
    print(f"  n_runs={info_numba['n_runs']}, n_hits={info_numba['n_hits']}, "
          f"no_hit_rate={info_numba['no_hit_rate']:.4f}")
    assert info_numba["same_seed_identical"]
    assert info_numba["diff_seed_different"]

    # ---- 2. Simulator reproducibility (Python)
    banner("2. Simulator reproducibility (Python backend)")
    info_python = _check_simulator_backend(seed_master, use_numba=False)
    print(f"  backend             = {info_python['backend']}")
    print(f"  same seed identical = {info_python['same_seed_identical']}")
    print(f"  diff seed different = {info_python['diff_seed_different']}")
    print(f"  n_runs={info_python['n_runs']}, n_hits={info_python['n_hits']}, "
          f"no_hit_rate={info_python['no_hit_rate']:.4f}")
    assert info_python["same_seed_identical"]
    assert info_python["diff_seed_different"]

    # ---- 3. IG MLE on synthetic IG data
    banner("3. IG MLE on synthetic IG data")
    mu_true, eta_true = 100.0, 50.0
    x = sample_inverse_gaussian(mu_true, eta_true, n=200_000, seed=seed_master)
    mu_hat, eta_hat = ig_mle_closed_form(x)
    rel_mu = abs(mu_hat - mu_true) / mu_true
    rel_eta = abs(eta_hat - eta_true) / eta_true
    print(f"  mu_true={mu_true}, mu_hat={mu_hat:.4f}, rel_err={rel_mu:.3e}")
    print(f"  eta_true={eta_true}, eta_hat={eta_hat:.4f}, rel_err={rel_eta:.3e}")
    assert rel_mu < 0.05
    assert rel_eta < 0.05

    # ---- 4. Beta log-linear fit on synthetic exact data
    banner("4. Beta log-linear fit (synthetic, exact)")
    a_true, b_true, c_true = 0.75, 1.10, 0.04
    p_grid = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    delta_grid = [1.0, 3.0, 5.0, 8.0, 12.0, 15.0]
    C_val, X0_val = 100, 50
    points = []
    for p in p_grid:
        for d in delta_grid:
            beta = a_true * (p ** (-b_true)) * np.exp(-c_true * d * np.log(p))
            lam = (1 - p) * C_val - d
            sigma_eff = lam + beta * p * (1 - p) * C_val ** 2
            eta = X0_val ** 2 / sigma_eff
            points.append(BetaPoint(
                p=p, delta=d, lam=lam, C=C_val, X0=X0_val,
                n_runs=15000, n_hits=14000,
                no_hit_rate=1.0 - 14000 / 15000,
                mu_hat=1.0, eta_hat=eta,
                sigma_eff_implied=sigma_eff, beta_implied=beta,
            ))
    fit = fit_beta_log_model(points, min_hits=500)
    print(f"  a_true={a_true},  a_hat={fit.a:.6f}  "
          f"(rel_err={abs(fit.a - a_true) / a_true:.3e})")
    print(f"  b_true={b_true},  b_hat={fit.b:.6f}  "
          f"(rel_err={abs(fit.b - b_true) / b_true:.3e})")
    print(f"  c_true={c_true},  c_hat={fit.c:.6f}  "
          f"(rel_err={abs(fit.c - c_true) / c_true:.3e})")
    print(f"  R^2 (log) = {fit.r2_log:.6f}")
    print(f"  R^2 (raw) = {fit.r2_raw:.6f}")
    print(f"  n_points_used / total = "
          f"{fit.n_points_used} / {fit.n_points_total}")
    assert abs(fit.a - a_true) / a_true < 0.10
    assert abs(fit.b - b_true) / b_true < 0.10
    assert abs(fit.c - c_true) / max(abs(c_true), 1e-12) < 0.10
    assert fit.r2_log > 0.99

    # ---- 5. Hazard diagnostics on a known DFR distribution
    banner("5. Hazard DFR on mixture of exponentials")
    t = np.arange(0, 2000, dtype=float)
    S_mix = 0.7 * np.exp(-0.005 * t) + 0.3 * np.exp(-0.05 * t)
    diag = hazard_diagnostics(S_mix, window=25, survival_tol=1e-8)
    print(f"  DFR_strict_raw         = {diag.DFR_strict_raw}")
    print(f"  DFR_strict_smoothed    = {diag.DFR_strict_smoothed}")
    print(f"  DFR_approx_smoothed    = {diag.DFR_approx_smoothed}")
    print(f"  frac_decreasing_smooth = {diag.frac_decreasing_smoothed:.6f}")
    print(f"  h_initial / h_final    = "
          f"{diag.h_initial:.6e} / {diag.h_final:.6e}")
    print(f"  n_hazard_points        = {diag.n_hazard_points}")
    assert diag.DFR_strict_raw is True
    assert diag.DFR_strict_smoothed is True
    assert diag.h_initial > diag.h_final

    # ---- 6. Persist artefact
    banner("6. Writing phase 3 artefacts")
    out_dir = ensure_dir(ROOT / "results" / "phase3")
    artefact = {
        "metadata": make_metadata(seed=seed_master),
        "simulator_numba": info_numba,
        "simulator_python": info_python,
        "ig_mle": {
            "mu_true": mu_true, "mu_hat": float(mu_hat),
            "rel_err_mu": float(rel_mu),
            "eta_true": eta_true, "eta_hat": float(eta_hat),
            "rel_err_eta": float(rel_eta),
        },
        "beta_fit_synthetic": {
            "a_true": a_true, "a_hat": fit.a, "se_a": fit.se_a,
            "b_true": b_true, "b_hat": fit.b, "se_b": fit.se_b,
            "c_true": c_true, "c_hat": fit.c, "se_c": fit.se_c,
            "r2_log": fit.r2_log, "r2_raw": fit.r2_raw,
            "mean_rel_error_eta": fit.mean_rel_error_eta,
            "max_rel_error_eta": fit.max_rel_error_eta,
            "n_points_used": fit.n_points_used,
            "n_points_total": fit.n_points_total,
            "n_discard_low_hits": fit.n_discard_low_hits,
            "n_discard_beta_nonpositive": fit.n_discard_beta_nonpositive,
        },
        "hazard_dfr_mixture": {
            "DFR_strict_raw": bool(diag.DFR_strict_raw),
            "DFR_strict_smoothed": bool(diag.DFR_strict_smoothed),
            "DFR_approx_smoothed": bool(diag.DFR_approx_smoothed),
            "frac_decreasing_smoothed": float(diag.frac_decreasing_smoothed),
            "h_initial": float(diag.h_initial),
            "h_final": float(diag.h_final),
            "n_hazard_points": int(diag.n_hazard_points),
        },
    }
    out_path = out_dir / "phase3_summary.json"
    save_json(out_path, artefact)
    print(f"  wrote {out_path}")

    # ---- 7. pytest
    banner("7. pytest")
    rc = subprocess.call(
        [
            sys.executable, "-m", "pytest",
            "tests/test_simulation.py",
            "tests/test_beta.py",
            "tests/test_hazard.py",
            "tests/test_reproducibility.py",
            "-q",
        ],
        cwd=str(ROOT),
    )
    if rc != 0:
        print("\n*** pytest FAILED ***")
        return rc

    banner("Phase 3: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
