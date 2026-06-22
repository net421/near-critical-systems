"""
Phase 2 validation script.

Checks:
1. RVI converges in <2000 iterations on baseline.
2. Threshold optimization on baseline (K_f=100, K_p=10) yields s* in [30, 45].
3. Full-action RVI returns a threshold policy.
4. Lemma 1 audit confirms s_star_wide < s_reset on baseline.
5. RVI cost agrees with threshold-search cost (rel < 1e-6).
6. Sanity: g(s*) for s* >= s_reset equals K_p exactly.
7. Persist phase 2 artefact JSON.
8. Run pytest.

Usage: python scripts/run_phase_2.py
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

from buffer_policy.io_utils import ensure_dir, make_metadata, save_json
from buffer_policy.kernel import build_kernel
from buffer_policy.mdp import (
    evaluate_threshold_policy,
    optimize_threshold,
    run_rvi,
)
from buffer_policy.params import baseline_cost, baseline_model
from buffer_policy.validation import audit_lemma1, audit_threshold_vs_rvi


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print(s)
    print("=" * 72)


def main() -> int:
    banner("Phase 2 validation")
    mp = baseline_model()
    cp = baseline_cost()
    print(f"  ModelParams: p={mp.p}, C={mp.C}, lam={mp.lam}, "
          f"S_max={mp.S_max}, X0={mp.X0}, s_reset={mp.s_reset}")
    print(f"  CostParams:  K_f={cp.K_f}, K_p={cp.K_p}")

    P = build_kernel(mp)

    # ---- 1. RVI convergence
    banner("1. RVI convergence")
    rvi = run_rvi(mp, cp, P=P, tol=1e-11, max_iter=5000)
    print(f"  converged       = {rvi.converged}")
    print(f"  iterations      = {rvi.n_iter}")
    print(f"  final span(diff)= {rvi.final_span:.3e}")
    print(f"  g (avg cost)    = {rvi.g:.6f}")
    assert rvi.converged
    assert rvi.n_iter < 2000

    # ---- 2. Threshold optimization
    banner("2. Threshold optimization")
    s_star, g_star, gs = optimize_threshold(mp, cp, P=P)
    print(f"  s*    = {s_star}")
    print(f"  g(s*) = {g_star:.6f}")
    print(f"  range searched: [0, {mp.s_reset - 1}]")
    print(f"  cost curve min/max = {gs.min():.6f} / {gs.max():.6f}")
    assert 30 <= s_star <= 45

    # ---- 3. Full-action RVI is threshold
    banner("3. Full-action RVI returns threshold policy")
    audit = audit_threshold_vs_rvi(mp, cp, P=P, rvi=rvi)
    print(f"  RVI policy is threshold? {audit.rvi_is_threshold}")
    print(f"  RVI s*                  = {audit.rvi_s_star}")
    print(f"  RVI cost                = {audit.g_rvi:.6f}")
    print(f"  threshold-search cost   = {audit.g_star_search:.6f}")
    print(f"  rel diff                = {audit.cost_rel_diff:.3e}")
    assert audit.rvi_is_threshold
    assert audit.cost_rel_diff < 1e-6

    # ---- 4. Lemma 1 audit
    banner("4. Lemma 1 audit (wide threshold search)")
    lem = audit_lemma1(mp, cp, P=P)
    print(f"  g_RTF        = {lem.g_RTF:.6f}")
    print(f"  K_p          = {lem.K_p}")
    print(f"  premise (g_RTF < K_p)? {lem.premise_holds}")
    print(f"  s_star_wide  = {lem.s_star_wide} (search [0, {mp.S_max}])")
    print(f"  g_star_wide  = {lem.g_star_wide:.6f}")
    print(f"  s_reset      = {lem.s_reset}")
    print(f"  holds        = {lem.holds}")
    assert lem.premise_holds
    assert lem.holds
    assert lem.s_star_wide < lem.s_reset

    # ---- 5. Cost consistency
    banner("5. Cost consistency RVI vs threshold search")
    rel = abs(rvi.g - g_star) / g_star
    print(f"  |g_rvi - g_star|/g_star = {rel:.3e}")
    assert rel < 1e-6

    # ---- 6. Wide-range sanity
    banner("6. Sanity: g(s*) = K_p for s* >= s_reset")
    for s_test in (mp.s_reset, mp.s_reset + 5, mp.S_max):
        g_test = evaluate_threshold_policy(s_test, mp, cp, P)
        print(f"  g(s*={s_test:3d}) = {g_test:.6f}  (expected {cp.K_p})")
        assert abs(g_test - cp.K_p) < 1e-12

    # ---- 7. Persist artefact
    banner("7. Writing phase 2 artefacts")
    out_dir = ensure_dir("results/phase2")
    save_json(
        out_dir / "phase2_summary.json",
        {
            "metadata": make_metadata(seed=None, model=mp, cost=cp),
            "rvi": {
                "converged": rvi.converged,
                "n_iter": rvi.n_iter,
                "final_span": rvi.final_span,
                "g": rvi.g,
                "policy_s_star": int(audit.rvi_s_star)
                                  if audit.rvi_s_star is not None else None,
            },
            "threshold_search": {
                "s_star": int(s_star),
                "g_star": g_star,
                "range": [0, mp.s_reset - 1],
            },
            "lemma1": {
                "g_RTF": lem.g_RTF,
                "K_p": lem.K_p,
                "premise_holds": lem.premise_holds,
                "s_star_wide": int(lem.s_star_wide),
                "g_star_wide": lem.g_star_wide,
                "s_reset": int(lem.s_reset),
                "holds": lem.holds,
            },
            "rvi_vs_search_rel_diff": rel,
        },
    )
    print(f"  wrote {out_dir / 'phase2_summary.json'}")

    # ---- 8. pytest
    banner("8. pytest")
    rc = subprocess.call([
        sys.executable, "-m", "pytest",
        "tests/test_mdp.py",
        "tests/test_validation.py",
        "-q",
    ])
    if rc != 0:
        print("\n*** pytest FAILED ***")
        return rc

    banner("Phase 2: ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
