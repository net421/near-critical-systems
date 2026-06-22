#!/usr/bin/env python
"""
Phase 4 driver: run all experiments and persist artefacts.

Pipeline:
  1. Cost ratio sweep        -> exp_cost_ratio.csv
  2. Heatmap (p x delta)     -> exp_heatmap.csv
  3. S_max sweep             -> exp_smax.csv
  4. Robustness scenarios    -> exp_robustness.csv
  5. Hazard diagnostics      -> diagnostic_hazard.csv
  6. Beta calibration (opt)  -> exp_beta.csv + beta_fit.json
  7. Plots                   -> figures/*.pdf
  8. Final summary           -> final_summary.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from buffer_policy.experiments import (  # noqa: E402
    run_beta_calibration_experiment,
    run_cost_ratio_experiment,
    run_hazard_diagnostics_experiment,
    run_heatmap_experiment,
    run_robustness_experiment,
    run_smax_experiment,
)
from buffer_policy.io_utils import (  # noqa: E402
    ensure_dir,
    make_metadata,
    save_json,
)
from buffer_policy.kernel import build_kernel  # noqa: E402
from buffer_policy.mdp import optimize_threshold  # noqa: E402
from buffer_policy.params import CostParams, ModelParams  # noqa: E402
from buffer_policy.plots import make_all_plots  # noqa: E402
from buffer_policy.survival import (  # noqa: E402
    expected_tau_linear_system,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Phase 4 driver")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--out", type=str, default="results/phase_4")
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--n-runs", type=int, default=15000)
    ap.add_argument("--max-steps", type=int, default=300_000)
    ap.add_argument("--skip-beta", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    return ap.parse_args(argv)


def baseline_summary() -> dict:
    """Compute baseline summary using capped deterministic MDP."""
    mp = ModelParams(p=0.15, lam=80.0, C=100, S_max=180, X0=50, s_reset=50)
    cp = CostParams(K_f=100.0, K_p=10.0)
    P = build_kernel(mp)
    E_tau = expected_tau_linear_system(P, mp)
    g_RTF = cp.K_f / E_tau
    s_star, g_star, _ = optimize_threshold(mp, cp, P=P)
    return {
        "p": mp.p, "lam": mp.lam, "C": mp.C, "S_max": mp.S_max,
        "X0": mp.X0, "s_reset": mp.s_reset,
        "K_f": cp.K_f, "K_p": cp.K_p,
        "E_tau": float(E_tau), "g_RTF": float(g_RTF),
        "s_star": int(s_star), "g_star": float(g_star),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out_root = Path(args.out)
    paper_dir = ensure_dir(out_root / "paper")
    fig_dir = ensure_dir(out_root / "figures")

    print("=" * 72)
    print(f"Phase 4 driver  mode={args.mode}  seed={args.seed}")
    print(f"  out_root  = {out_root}")
    print(f"  skip_beta = {args.skip_beta}")
    print(f"  no_plots  = {args.no_plots}")
    print("=" * 72)

    timing: dict[str, float] = {}
    files: dict[str, str] = {}
    t_total = time.time()

    print("\n[phase4] baseline summary")
    t0 = time.time()
    baseline = baseline_summary()
    timing["baseline"] = time.time() - t0
    print(f"  s_star={baseline['s_star']}, g_star={baseline['g_star']:.6f}, "
          f"E_tau={baseline['E_tau']:.4f}")

    # 1. cost ratio
    print("\n[phase4] (1/6) cost ratio sweep")
    t0 = time.time()
    df_cost = run_cost_ratio_experiment(args.mode)
    p_cost = paper_dir / "exp_cost_ratio.csv"
    df_cost.to_csv(p_cost, index=False)
    files["exp_cost_ratio"] = str(p_cost)
    timing["cost_ratio"] = time.time() - t0
    print(f"  rows={len(df_cost)}  wrote {p_cost}  "
          f"({timing['cost_ratio']:.2f}s)")

    # 2. heatmap
    print("\n[phase4] (2/6) heatmap (p x delta)")
    t0 = time.time()
    df_heat = run_heatmap_experiment(args.mode)
    p_heat = paper_dir / "exp_heatmap.csv"
    df_heat.to_csv(p_heat, index=False)
    files["exp_heatmap"] = str(p_heat)
    timing["heatmap"] = time.time() - t0
    print(f"  rows={len(df_heat)}  valid={int(df_heat['valid'].sum())}  "
          f"wrote {p_heat}  ({timing['heatmap']:.2f}s)")

    # 3. S_max sweep
    print("\n[phase4] (3/6) S_max sweep")
    t0 = time.time()
    df_smax = run_smax_experiment(args.mode)
    p_smax = paper_dir / "exp_smax.csv"
    df_smax.to_csv(p_smax, index=False)
    files["exp_smax"] = str(p_smax)
    timing["smax"] = time.time() - t0
    print(f"  rows={len(df_smax)}  valid={int(df_smax['valid'].sum())}  "
          f"wrote {p_smax}  ({timing['smax']:.2f}s)")

    # 4. robustness
    print("\n[phase4] (4/6) robustness scenarios")
    t0 = time.time()
    df_rob = run_robustness_experiment(args.mode)
    p_rob = paper_dir / "exp_robustness.csv"
    df_rob.to_csv(p_rob, index=False)
    files["exp_robustness"] = str(p_rob)
    timing["robustness"] = time.time() - t0
    print(f"  rows={len(df_rob)}  valid={int(df_rob['valid'].sum())}  "
          f"wrote {p_rob}  ({timing['robustness']:.2f}s)")

    # 5. hazard
    print("\n[phase4] (5/6) hazard diagnostics (capped)")
    t0 = time.time()
    df_haz = run_hazard_diagnostics_experiment(args.mode)
    p_haz = paper_dir / "diagnostic_hazard.csv"
    df_haz.to_csv(p_haz, index=False)
    files["diagnostic_hazard"] = str(p_haz)
    timing["hazard"] = time.time() - t0
    print(f"  rows={len(df_haz)}  valid={int(df_haz['valid'].sum())}  "
          f"wrote {p_haz}  ({timing['hazard']:.2f}s)")

    # 6. beta (optional)
    beta_fit_dict: dict | None = None
    if args.skip_beta:
        print("\n[phase4] (6/6) beta calibration  SKIPPED (--skip-beta)")
        timing["beta"] = 0.0
    else:
        print("\n[phase4] (6/6) beta calibration (uncapped MC)")
        t0 = time.time()
        df_beta, beta_fit_dict = run_beta_calibration_experiment(
            mode=args.mode,
            n_runs=args.n_runs,
            max_steps=args.max_steps,
            seed=args.seed,
        )
        p_beta = paper_dir / "exp_beta.csv"
        df_beta.to_csv(p_beta, index=False)
        files["exp_beta"] = str(p_beta)
        if beta_fit_dict is not None:
            p_fit = paper_dir / "beta_fit.json"
            save_json(p_fit, beta_fit_dict)
            files["beta_fit"] = str(p_fit)
        timing["beta"] = time.time() - t0
        print(f"  rows={len(df_beta)}  wrote {p_beta}  "
              f"({timing['beta']:.2f}s)")
        if beta_fit_dict is not None and "error" not in beta_fit_dict:
            print(f"  fit: a={beta_fit_dict.get('a'):.4f}, "
                  f"b={beta_fit_dict.get('b'):.4f}, "
                  f"c={beta_fit_dict.get('c'):.5f}, "
                  f"R2_log={beta_fit_dict.get('r2_log'):.4f}")
        elif beta_fit_dict is not None:
            print(f"  fit: ERROR -> {beta_fit_dict.get('error')}")

    # 7. plots
    figs: list[Path] = []
    if args.no_plots:
        print("\n[phase4] plots SKIPPED (--no-plots)")
        timing["plots"] = 0.0
    else:
        print("\n[phase4] generating plots")
        t0 = time.time()
        try:
            figs = make_all_plots(paper_dir, fig_dir)
            for f in figs:
                print(f"  wrote {f}")
        except Exception as exc:
            print(f"  WARNING: plotting failed: {type(exc).__name__}: {exc}")
        timing["plots"] = time.time() - t0

    # 8. final summary
    elapsed = time.time() - t_total
    summary = {
        "metadata": make_metadata(seed=args.seed),
        "config": {
            "mode": args.mode, "seed": args.seed,
            "n_runs": args.n_runs, "max_steps": args.max_steps,
            "skip_beta": bool(args.skip_beta),
            "no_plots": bool(args.no_plots),
            "out_root": str(out_root),
        },
        "baseline": baseline,
        "experiments": {
            "cost_ratio_rows": int(len(df_cost)),
            "heatmap_rows": int(len(df_heat)),
            "heatmap_valid": int(df_heat["valid"].sum()),
            "smax_rows": int(len(df_smax)),
            "smax_valid": int(df_smax["valid"].sum()),
            "robustness_rows": int(len(df_rob)),
            "robustness_valid": int(df_rob["valid"].sum()),
            "hazard_rows": int(len(df_haz)),
            "hazard_valid": int(df_haz["valid"].sum()),
        },
        "beta_skipped": bool(args.skip_beta),
        "beta_fit": beta_fit_dict,
        "files": {
            "csvs": files,
            "figures": [str(f) for f in figs],
        },
        "timing_seconds": {**timing, "total": elapsed},
    }
    summary_path = out_root / "final_summary.json"
    save_json(summary_path, summary)
    print(f"\n[phase4] wrote {summary_path}")
    print(f"[phase4] total elapsed: {elapsed:.2f}s")
    print("[phase4] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
