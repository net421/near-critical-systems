"""
Phase 1 validation script.

Checks:
1. Build kernel for baseline; row sums within 1e-12 of 1.
2. Compare row at s=90 against scipy reference; max |diff| < 1e-14.
3. Survival: S(0)=1, monotone non-increasing.
4. E[tau] iterative vs linear-system audit (rel diff < 1e-6).
   g_RTF = K_f / E[tau] is finite, positive, < K_f.
5. Age-based cost g_AB(T) at T=100 finite; g_AB(5000) within 0.1% of g_RTF.
6. Persist phase 1 artefact JSON.
7. Run pytest.

Usage: python scripts/run_phase_1.py
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
from scipy.stats import poisson

from buffer_policy.age_based import age_based_cost
from buffer_policy.io_utils import ensure_dir, make_metadata, save_json
from buffer_policy.kernel import build_kernel
from buffer_policy.params import baseline_cost, baseline_model
from buffer_policy.survival import (
    expected_tau,
    expected_tau_linear_system,
    g_run_to_failure,
    g_run_to_failure_linear_system,
    survival_from_kernel,
)


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def main() -> int:
    banner("Phase 1 validation")
    mp = baseline_model()
    cp = baseline_cost()
    print(f"  ModelParams: p={mp.p}, C={mp.C}, lam={mp.lam}, "
          f"S_max={mp.S_max}, X0={mp.X0}, s_reset={mp.s_reset}")
    print(f"  delta = {mp.delta:.4f}, sigma2 = {mp.sigma2:.4f}, "
          f"u = {mp.utilization:.4f}")
    print(f"  CostParams:  K_f={cp.K_f}, K_p={cp.K_p}")

    # ---- 1. Kernel & row sums
    banner("1. Kernel construction & row-sum audit")
    P = build_kernel(mp, audit_tol=1e-12)
    rs = P.sum(axis=1)
    err = float(np.max(np.abs(rs - 1.0)))
    print(f"  shape = {P.shape}, max |row_sum - 1| = {err:.3e}")
    print(f"  min entry = {P.min():.3e}, max entry = {P.max():.6f}")
    assert err < 1e-12
    assert P.min() >= 0.0

    # ---- 2. Row at s=90 matches scipy
    banner("2. Row at s=90 vs scipy reference")
    s = 90
    expected = np.zeros(mp.S_max + 1)
    expected[0] = (
        (1 - mp.p) * poisson.sf(s + mp.C - 1, mu=mp.lam)
        + mp.p * poisson.sf(s - 1, mu=mp.lam)
    )
    for sp in range(1, s + 1):
        expected[sp] += mp.p * poisson.pmf(s - sp, mu=mp.lam)
    for sp in range(1, mp.S_max):
        expected[sp] += (1 - mp.p) * poisson.pmf(s + mp.C - sp, mu=mp.lam)
    m = s + mp.C - mp.S_max
    if m >= 0:
        expected[mp.S_max] += (1 - mp.p) * poisson.cdf(m, mu=mp.lam)
    diff = float(np.max(np.abs(P[s] - expected)))
    print(f"  max |P[90,:] - reference| = {diff:.3e}")
    assert diff < 1e-14

    # ---- 3. Survival
    banner("3. Survival distribution")
    S = survival_from_kernel(P, mp, T_max=500_000, tol=1e-20)
    print(f"  len(S) = {len(S)},  S[0] = {S[0]:.6f},  S[-1] = {S[-1]:.3e}")
    assert S[0] == 1.0
    assert np.all(np.diff(S) <= 1e-15)

    # ---- 4. E[tau] iterative vs linear-system audit
    banner("4. E[tau] iterative vs linear-system audit")
    Etau = expected_tau(S)
    Etau_lin = expected_tau_linear_system(P, mp)
    rel_etau = abs(Etau - Etau_lin) / Etau_lin
    g_rtf = g_run_to_failure(S, cp.K_f)
    g_rtf_lin = g_run_to_failure_linear_system(P, mp, cp.K_f)
    rel_g = abs(g_rtf - g_rtf_lin) / g_rtf_lin

    print(f"  E[tau] (iterative)     = {Etau:.6f}")
    print(f"  E[tau] (linear system) = {Etau_lin:.6f}")
    print(f"  rel diff E[tau]        = {rel_etau:.3e}")
    print(f"  g_RTF  (iterative)     = {g_rtf:.6f}")
    print(f"  g_RTF  (linear system) = {g_rtf_lin:.6f}")
    print(f"  rel diff g_RTF         = {rel_g:.3e}")
    assert np.isfinite(Etau) and Etau > 1.0
    assert np.isfinite(Etau_lin) and Etau_lin > 1.0
    assert rel_etau < 1e-6
    assert rel_g < 1e-6
    assert 0.0 < g_rtf < cp.K_f

    # ---- 5. Age-based cost
    banner("5. Age-based cost")
    g_ab_100 = age_based_cost(S, 100, cp.K_f, cp.K_p)
    T_far = min(len(S) - 1, 5000)
    g_ab_far = age_based_cost(S, T_far, cp.K_f, cp.K_p)
    rel_ab = abs(g_ab_far - g_rtf_lin) / g_rtf_lin
    print(f"  g_AB(100)            = {g_ab_100:.6f}")
    print(f"  g_AB({T_far})          = {g_ab_far:.6f}")
    print(f"  rel diff vs g_RTF    = {rel_ab:.3e}")
    assert np.isfinite(g_ab_100) and g_ab_100 > 0.0
    assert rel_ab < 1e-3

    # ---- 6. Persist artefact
    banner("6. Writing phase 1 artefacts")
    out_dir = ensure_dir("results/phase1")
    save_json(
        out_dir / "phase1_summary.json",
        {
            "metadata": make_metadata(seed=None, model=mp, cost=cp),
            "kernel_row_sum_err": err,
            "row90_max_abs_diff": diff,
            "len_survival": int(len(S)),
            "E_tau_iterative": Etau,
            "E_tau_linear_system": Etau_lin,
            "rel_diff_E_tau": rel_etau,
            "g_RTF_iterative": g_rtf,
            "g_RTF_linear_system": g_rtf_lin,
            "rel_diff_g_RTF": rel_g,
            "g_AB_100": g_ab_100,
            f"g_AB_{T_far}": g_ab_far,
            "rel_diff_g_AB_vs_RTF": rel_ab,
        },
    )
    print(f"  wrote {out_dir / 'phase1_summary.json'}")

    # ---- 7. pytest
    banner("7. pytest")
    rc = subprocess.call([
        sys.executable, "-m", "pytest",
        "tests/test_poisson.py",
        "tests/test_kernel.py",
        "tests/test_survival_agebased.py",
        "-q",
    ])
    if rc != 0:
        print("\n*** pytest FAILED ***")
        return rc

    banner("Phase 1: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
