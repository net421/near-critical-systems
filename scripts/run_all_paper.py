#!/usr/bin/env python
"""
Paper-grade reproduction driver.

Pipeline:
  1. Baseline               -> baseline dict in summary
  2. Beta calibration (opt) -> exp_beta_points_paper.csv
                             + exp_beta_validity_paper.csv
                             + beta_fit_paper.json
  3. Cost ratio (Table 4)   -> exp_cost_ratio_paper.csv
  4. Heatmap (Tables 5/8)   -> exp_heatmap_paper.csv
  5. S_max (Table 6)        -> exp_smax_paper.csv
                             + exp_smax_summary_paper.csv
  6. Misspecification (T 7) -> exp_misspecification_paper.csv
  7. Hazard (Table 3)       -> diagnostic_hazard_paper.csv
                             + diagnostic_hazard_summary_paper.json
  8. Lemma 1 audit          -> lemma1_audit_paper.csv
  9. Plots (optional)       -> figures/*.pdf
 10. Final summary          -> final_summary_paper.json

CLI flags:
  --full                       (default unless --beta-quick)
  --beta-quick                 n_runs=3000, max_steps=50000
  --skip-beta                  skip beta MC entirely
  --allow-incomplete-full      required to combine --full with --skip-beta
  --out             default "results/full_paper"
  --seed            default 20260429
  --no-plots
  --no-numba-beta
  --legacy-beta-seeds          activates seed_mode="legacy_seed_i"
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from buffer_policy.experiments_paper import (  # noqa: E402
    hazard_summary,
    run_baseline_paper,
    run_cost_ratio_paper,
    run_hazard_paper,
    run_heatmap_paper,
    run_lemma1_audit_paper,
    run_misspecification_paper,
    run_smax_paper,
    smax_summary,
)
from buffer_policy.ig_beta_paper import (  # noqa: E402
    run_beta_calibration_paper,
)
from buffer_policy.io_utils import (  # noqa: E402
    ensure_dir,
    make_metadata,
    save_json,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Paper-grade driver")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true",
                      help="Full paper config (default unless --beta-quick).")
    mode.add_argument("--beta-quick", action="store_true",
                      help="Quick beta calibration (n_runs=3000).")
    ap.add_argument("--skip-beta", action="store_true")
    ap.add_argument("--allow-incomplete-full", action="store_true")
    ap.add_argument("--out", type=str, default="results/full_paper")
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-numba-beta", action="store_true")
    ap.add_argument("--legacy-beta-seeds", action="store_true")
    return ap.parse_args(argv)


def banner(s: str) -> None:
    print("\n" + "=" * 72)
    print("[run_all_paper] " + s)
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Default to --full unless --beta-quick
    is_beta_quick = bool(args.beta_quick)
    is_full = not is_beta_quick

    # Guard: --full + --skip-beta requires explicit override
    if is_full and args.skip_beta and not args.allow_incomplete_full:
        raise ValueError(
            "--full does not allow --skip-beta unless "
            "--allow-incomplete-full is set."
        )

    # beta config
    if is_beta_quick:
        beta_n_runs = 3000
        beta_max_steps = 50_000
    else:
        beta_n_runs = 15_000
        beta_max_steps = 300_000

    seed_mode = "legacy_seed_i" if args.legacy_beta_seeds else "spawn"
    use_numba_beta = not args.no_numba_beta

    out_root = Path(args.out)
    paper_dir = ensure_dir(out_root / "paper")
    fig_dir = ensure_dir(out_root / "figures")

    banner("config")
    print(f"  mode             = {'beta_quick' if is_beta_quick else 'full'}")
    print(f"  out_root         = {out_root}")
    print(f"  seed             = {args.seed}")
    print(f"  skip_beta        = {args.skip_beta}")
    print(f"  no_plots         = {args.no_plots}")
    print(f"  no_numba_beta    = {args.no_numba_beta}")
    print(f"  legacy_beta_seeds= {args.legacy_beta_seeds}")
    print(f"  beta n_runs      = {beta_n_runs}")
    print(f"  beta max_steps   = {beta_max_steps}")

    timing: dict[str, float] = {}
    files: dict[str, str] = {}
    t_total = time.time()

    # 1. baseline
    banner("(1/8) baseline")
    t0 = time.time()
    baseline = run_baseline_paper()
    timing["baseline"] = time.time() - t0
    print(f"  s_star={baseline['s_star']}, "
          f"g_star={baseline['g_star']:.6f}, "
          f"E_tau={baseline['E_tau']:.4f}")

    # 2. beta calibration (optional)
    beta_fit_dict: dict | None = None
    if args.skip_beta:
        banner("(2/8) beta calibration  SKIPPED (--skip-beta)")
        timing["beta"] = 0.0
    else:
        banner("(2/8) beta calibration (uncapped MC, paper-grade)")
        t0 = time.time()
        df_all, df_valid, beta_fit_dict = run_beta_calibration_paper(
            n_runs=beta_n_runs,
            max_steps=beta_max_steps,
            seed=args.seed,
            seed_mode=seed_mode,
            use_numba=use_numba_beta,
        )
        p_all = paper_dir / "exp_beta_points_paper.csv"
        p_valid = paper_dir / "exp_beta_validity_paper.csv"
        df_all.to_csv(p_all, index=False)
        df_valid.to_csv(p_valid, index=False)
        files["exp_beta_points_paper"] = str(p_all)
        files["exp_beta_validity_paper"] = str(p_valid)
        if beta_fit_dict is not None:
            p_fit = paper_dir / "beta_fit_paper.json"
            save_json(p_fit, beta_fit_dict)
            files["beta_fit_paper"] = str(p_fit)
        timing["beta"] = time.time() - t0
        print(f"  rows_all={len(df_all)}  rows_valid={len(df_valid)}  "
              f"({timing['beta']:.2f}s)")
        if beta_fit_dict is not None and "error" not in beta_fit_dict:
            print(f"  fit: a={beta_fit_dict.get('a'):.4f}, "
                  f"b={beta_fit_dict.get('b'):.4f}, "
                  f"c={beta_fit_dict.get('c'):.5f}, "
                  f"R2_raw={beta_fit_dict.get('R2_raw_beta'):.4f}")
            print(f"  MAE eta uncorrected = "
                  f"{beta_fit_dict.get('MAE_uncorrected_eta_pct'):.2f}%")
            print(f"  MAE eta corrected   = "
                  f"{beta_fit_dict.get('MAE_corrected_eta_pct'):.2f}%")
            print(f"  improvement factor  = "
                  f"{beta_fit_dict.get('improvement_factor'):.2f}x")
        elif beta_fit_dict is not None:
            print(f"  fit: ERROR -> {beta_fit_dict.get('error')}")

    # 3. cost ratio (Table 4)
    banner("(3/8) cost ratio sweep (Table 4)")
    t0 = time.time()
    df_cost = run_cost_ratio_paper()
    p_cost = paper_dir / "exp_cost_ratio_paper.csv"
    df_cost.to_csv(p_cost, index=False)
    files["exp_cost_ratio_paper"] = str(p_cost)
    timing["cost_ratio"] = time.time() - t0
    print(f"  rows={len(df_cost)}  ({timing['cost_ratio']:.2f}s)")

    # 4. heatmap (Tables 5 & 8)
    banner("(4/8) heatmap p x util (Tables 5 & 8)")
    t0 = time.time()
    df_heat = run_heatmap_paper()
    p_heat = paper_dir / "exp_heatmap_paper.csv"
    df_heat.to_csv(p_heat, index=False)
    files["exp_heatmap_paper"] = str(p_heat)
    timing["heatmap"] = time.time() - t0
    print(f"  rows={len(df_heat)}  valid={int(df_heat['valid'].sum())}  "
          f"({timing['heatmap']:.2f}s)")

    # 5. S_max sweep (Table 6)
    banner("(5/8) S_max sweep (Table 6)")
    t0 = time.time()
    df_smax = run_smax_paper()
    p_smax = paper_dir / "exp_smax_paper.csv"
    df_smax.to_csv(p_smax, index=False)
    files["exp_smax_paper"] = str(p_smax)
    df_smax_sum = smax_summary(df_smax)
    p_smax_sum = paper_dir / "exp_smax_summary_paper.csv"
    df_smax_sum.to_csv(p_smax_sum, index=False)
    files["exp_smax_summary_paper"] = str(p_smax_sum)
    timing["smax"] = time.time() - t0
    print(f"  rows={len(df_smax)}  valid={int(df_smax['valid'].sum())}  "
          f"({timing['smax']:.2f}s)")

    # 6. misspecification (Table 7)
    banner("(6/8) misspecification (Table 7)")
    t0 = time.time()
    df_mis = run_misspecification_paper()
    p_mis = paper_dir / "exp_misspecification_paper.csv"
    df_mis.to_csv(p_mis, index=False)
    files["exp_misspecification_paper"] = str(p_mis)
    timing["misspec"] = time.time() - t0
    print(f"  rows={len(df_mis)}  valid={int(df_mis['valid'].sum())}  "
          f"({timing['misspec']:.2f}s)")

    # 7. hazard (Table 3)
    banner("(7/8) hazard diagnostics (Table 3)")
    t0 = time.time()
    df_haz = run_hazard_paper()
    p_haz = paper_dir / "diagnostic_hazard_paper.csv"
    df_haz.to_csv(p_haz, index=False)
    files["diagnostic_hazard_paper"] = str(p_haz)
    haz_sum = hazard_summary(df_haz)
    p_haz_sum = paper_dir / "diagnostic_hazard_summary_paper.json"
    save_json(p_haz_sum, haz_sum)
    files["diagnostic_hazard_summary_paper"] = str(p_haz_sum)
    timing["hazard"] = time.time() - t0
    print(f"  rows={len(df_haz)}  valid={int(df_haz['valid'].sum())}  "
          f"DFR_strict={haz_sum.get('DFR_strict_total')}/"
          f"{haz_sum.get('total_configs')}  "
          f"({timing['hazard']:.2f}s)")

    # 8. lemma 1 audit
    banner("(8/8) Lemma 1 audit")
    t0 = time.time()
    df_lem = run_lemma1_audit_paper()
    p_lem = paper_dir / "lemma1_audit_paper.csv"
    df_lem.to_csv(p_lem, index=False)
    files["lemma1_audit_paper"] = str(p_lem)
    timing["lemma1"] = time.time() - t0
    n_holds = int(df_lem[df_lem["valid"]]["lemma1_holds"].sum())
    n_valid = int(df_lem["valid"].sum())
    print(f"  valid={n_valid}  lemma1_holds={n_holds}/{n_valid}  "
          f"({timing['lemma1']:.2f}s)")

    # plots (optional, reuse plots.py)
    figs: list[Path] = []
    if args.no_plots:
        banner("plots SKIPPED (--no-plots)")
        timing["plots"] = 0.0
    else:
        banner("plots (reusing plots.py where possible)")
        t0 = time.time()
        try:
            from buffer_policy.plots import make_all_plots
            # plots.py expects modular CSV names; rename a temp dir
            # by symlinking equivalent files. To keep things simple,
            # we just copy the paper CSVs under modular names and
            # call make_all_plots over a tmp dir.
            import shutil
            tmp_dir = ensure_dir(out_root / "_plots_tmp")
            mapping = {
                "exp_cost_ratio_paper.csv": "exp_cost_ratio.csv",
                "exp_heatmap_paper.csv": "exp_heatmap.csv",
                "exp_smax_paper.csv": "exp_smax.csv",
                "diagnostic_hazard_paper.csv": "diagnostic_hazard.csv",
            }
            for src_name, dst_name in mapping.items():
                src = paper_dir / src_name
                if src.exists():
                    shutil.copy2(src, tmp_dir / dst_name)
            # plots.py also looks for exp_robustness; skip if absent
            figs = make_all_plots(tmp_dir, fig_dir)
            for f in figs:
                print(f"  wrote {f}")
        except Exception as exc:
            print(f"  WARNING: plotting failed: "
                  f"{type(exc).__name__}: {exc}")
        timing["plots"] = time.time() - t0

    # final summary
    elapsed = time.time() - t_total
    summary = {
        "metadata": make_metadata(seed=args.seed),
        "mode": ("beta_quick_reproduction" if is_beta_quick
                 else "full_paper_reproduction"),
        "config": {
            "full": bool(is_full),
            "beta_quick": bool(is_beta_quick),
            "skip_beta": bool(args.skip_beta),
            "allow_incomplete_full": bool(args.allow_incomplete_full),
            "seed": int(args.seed),
            "out_root": str(out_root),
            "no_plots": bool(args.no_plots),
            "no_numba_beta": bool(args.no_numba_beta),
            "legacy_beta_seeds": bool(args.legacy_beta_seeds),
            "beta_n_runs": int(beta_n_runs),
            "beta_max_steps": int(beta_max_steps),
        },
        "baseline": baseline,
        "experiments": {
            "cost_ratio_rows": int(len(df_cost)),
            "heatmap_rows": int(len(df_heat)),
            "heatmap_valid": int(df_heat["valid"].sum()),
            "smax_rows": int(len(df_smax)),
            "smax_valid": int(df_smax["valid"].sum()),
            "misspec_rows": int(len(df_mis)),
            "misspec_valid": int(df_mis["valid"].sum()),
            "hazard_rows": int(len(df_haz)),
            "hazard_valid": int(df_haz["valid"].sum()),
            "lemma1_rows": int(len(df_lem)),
            "lemma1_valid": int(df_lem["valid"].sum()),
        },
        "beta_skipped": bool(args.skip_beta),
        "beta_fit": beta_fit_dict,
        "hazard_summary": haz_sum,
        "files": {
            "csvs": files,
            "figures": [str(f) for f in figs],
        },
        "timing_seconds": {**timing, "total": elapsed},
    }
    summary_path = out_root / "final_summary_paper.json"
    save_json(summary_path, summary)
    print(f"\n[run_all_paper] wrote {summary_path}")
    print(f"[run_all_paper] total elapsed: {elapsed:.2f}s")
    print("[run_all_paper] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
