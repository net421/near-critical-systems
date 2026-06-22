#!/usr/bin/env python
"""
Smoke test: minimal end-to-end exercise of the full stack.

Runs a tiny version of every component to catch import / wiring bugs
without performing a real experiment. Exits 0 on success, 1 on any failure.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402

from buffer_policy.hazard import hazard_diagnostics  # noqa: E402
from buffer_policy.ig_beta import (  # noqa: E402
    BetaPoint,    fit_beta_log_model,
    ig_mle_closed_form,
    sample_inverse_gaussian,
)
from buffer_policy.kernel import build_kernel  # noqa: E402
from buffer_policy.mdp import optimize_threshold  # noqa: E402
from buffer_policy.params import CostParams, ModelParams  # noqa: E402
from buffer_policy.simulation import (  # noqa: E402
    simulate_uncapped_first_passage,
)
from buffer_policy.survival import (  # noqa: E402
    expected_tau_linear_system,
    survival_from_kernel,
)


def step(name: str, fn) -> bool:
    print(f"\n--- {name} ---")
    try:
        fn()
        print("  OK")
        return True
    except Exception as exc:
        print(f"  FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def step1_kernel_survival_etau() -> None:
    mp = ModelParams(p=0.15, lam=80.0, C=100, S_max=180, X0=50, s_reset=50)
    P = build_kernel(mp)
    E_tau = expected_tau_linear_system(P, mp)
    S = survival_from_kernel(P, mp, T_max=2000, tol=1e-10)
    assert np.isfinite(E_tau) and E_tau > 0.0, f"bad E_tau: {E_tau}"
    assert S.shape[0] >= 2 and S[0] == 1.0
    assert np.all(np.diff(S) <= 1e-12), "survival not monotone"
    print(f"  E[tau]={E_tau:.4f}, |S|={S.shape[0]}, S[-1]={S[-1]:.3e}")


def step2_threshold() -> None:
    mp = ModelParams(p=0.15, lam=80.0, C=100, S_max=180, X0=50, s_reset=50)
    cp = CostParams(K_f=100.0, K_p=10.0)
    s_star, g_star, _ = optimize_threshold(mp, cp)
    assert 0 <= s_star <= mp.S_max, f"s_star out of range: {s_star}"
    assert np.isfinite(g_star) and g_star > 0
    print(f"  s*={s_star}, g*={g_star:.4f}")


def step3_sim_numba() -> None:
    res = simulate_uncapped_first_passage(
        X0=10, C=20, p=0.2, lam=15.0,
        n_runs=200, max_steps=2000, seed=2024, use_numba=True,
    )
    assert res.backend == "numba"
    assert res.n_runs == 200
    assert res.taus.shape == (200,)
    print(f"  backend={res.backend}, n_hits={res.n_hits}, "
          f"no_hit_rate={res.no_hit_rate:.3f}")


def step4_sim_python() -> None:
    res = simulate_uncapped_first_passage(
        X0=10, C=20, p=0.2, lam=15.0,
        n_runs=200, max_steps=2000, seed=2024, use_numba=False,
    )
    assert res.backend == "python"
    assert res.n_runs == 200
    print(f"  backend={res.backend}, n_hits={res.n_hits}, "
          f"no_hit_rate={res.no_hit_rate:.3f}")


def step5_hazard() -> None:
    t = np.arange(0, 1000, dtype=float)
    S = 0.7 * np.exp(-0.005 * t) + 0.3 * np.exp(-0.05 * t)
    diag = hazard_diagnostics(S, window=25, survival_tol=1e-8)
    assert diag.DFR_strict_raw is True
    assert diag.h_initial > diag.h_final
    print(f"  DFR_strict_raw={diag.DFR_strict_raw}, "
          f"h_initial={diag.h_initial:.4e}, h_final={diag.h_final:.4e}")


def step6_beta_fit() -> None:
    # IG MLE smoke
    x = sample_inverse_gaussian(100.0, 50.0, n=20_000, seed=1)
    mu, eta = ig_mle_closed_form(x)
    assert abs(mu - 100.0) / 100.0 < 0.05
    assert abs(eta - 50.0) / 50.0 < 0.10

    # synthetic exact beta surface
    a, b, c = 0.8, 1.2, 0.05
    pts: list[BetaPoint] = []
    for p in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        for d in [1.0, 3.0, 5.0, 8.0, 12.0, 15.0]:
            beta = a * (p ** (-b)) * np.exp(-c * d * np.log(p))
            lam = (1 - p) * 100 - d
            sigma_eff = lam + beta * p * (1 - p) * 100 ** 2
            eta_pt = 50 ** 2 / sigma_eff
            pts.append(BetaPoint(
                p=p, delta=d, lam=lam, C=100, X0=50,
                n_runs=15000, n_hits=14000,
                no_hit_rate=1.0 - 14000 / 15000,
                mu_hat=1.0, eta_hat=eta_pt,
                sigma_eff_implied=sigma_eff, beta_implied=beta,
            ))
    fit = fit_beta_log_model(pts, min_hits=500)
    assert fit.r2_log > 0.99
    print(f"  a={fit.a:.4f}, b={fit.b:.4f}, c={fit.c:.5f}, "
          f"R2_log={fit.r2_log:.4f}")


def main() -> int:
    print("=" * 60)
    print("SMOKE TEST")
    print("=" * 60)
    results = [
        step("[1] kernel + survival + E[tau]", step1_kernel_survival_etau),
        step("[2] threshold optimization", step2_threshold),
        step("[3] simulator (numba)", step3_sim_numba),
        step("[4] simulator (python)", step4_sim_python),
        step("[5] hazard diagnostics", step5_hazard),
        step("[6] IG MLE + beta log fit", step6_beta_fit),
    ]
    print("\n" + "=" * 60)
    if all(results):
        print("SMOKE TEST: ALL PASS")
        return 0
    print(f"SMOKE TEST: {results.count(False)} FAILURES out of {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
