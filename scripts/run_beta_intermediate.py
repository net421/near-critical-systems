#!/usr/bin/env python
"""
Run intermediate paper-grade beta calibration only.

This avoids recomputing all deterministic paper tables.

Example:
  python scripts/run_beta_intermediate.py --n-runs 3000 --max-steps 150000
  python scripts/run_beta_intermediate.py --n-runs 3000 --max-steps 300000
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

from buffer_policy.ig_beta_paper import run_beta_calibration_paper
from buffer_policy.io_utils import ensure_dir, make_metadata, save_json


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Intermediate beta calibration")
    ap.add_argument("--n-runs", type=int, default=3000)
    ap.add_argument("--max-steps", type=int, default=150000)
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--out", type=str, default="results/beta_intermediate")
    ap.add_argument("--no-numba", action="store_true")
    ap.add_argument("--legacy-beta-seeds", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    out_root = ensure_dir(args.out)
    seed_mode = "legacy_seed_i" if args.legacy_beta_seeds else "spawn"
    use_numba = not args.no_numba

    print("=" * 72)
    print("INTERMEDIATE BETA CALIBRATION")
    print("=" * 72)
    print(f"  n_runs       = {args.n_runs}")
    print(f"  max_steps    = {args.max_steps}")
    print(f"  seed         = {args.seed}")
    print(f"  seed_mode    = {seed_mode}")
    print(f"  use_numba    = {use_numba}")
    print(f"  out_root     = {out_root}")
    print("=" * 72)

    t0 = time.time()

    df_all, df_valid, fit = run_beta_calibration_paper(
        n_runs=args.n_runs,
        max_steps=args.max_steps,
        seed=args.seed,
        seed_mode=seed_mode,
        use_numba=use_numba,
    )

    elapsed = time.time() - t0

    p_all = out_root / f"beta_points_n{args.n_runs}_T{args.max_steps}.csv"
    p_valid = out_root / f"beta_validity_n{args.n_runs}_T{args.max_steps}.csv"
    p_fit = out_root / f"beta_fit_n{args.n_runs}_T{args.max_steps}.json"
    p_summary = out_root / f"summary_n{args.n_runs}_T{args.max_steps}.json"

    df_all.to_csv(p_all, index=False)
    df_valid.to_csv(p_valid, index=False)
    save_json(p_fit, fit)

    summary = {
        "metadata": make_metadata(seed=args.seed),
        "config": {
            "n_runs": args.n_runs,
            "max_steps": args.max_steps,
            "seed": args.seed,
            "seed_mode": seed_mode,
            "use_numba": use_numba,
        },
        "rows_all": int(len(df_all)),
        "rows_valid": int(len(df_valid)),
        "fit": fit,
        "timing_seconds": elapsed,
        "files": {
            "all_points": str(p_all),
            "validity": str(p_valid),
            "fit": str(p_fit),
        },
    }
    save_json(p_summary, summary)

    print("\nRESULTS")
    print("-" * 72)
    print(f"  rows_all   = {len(df_all)}")
    print(f"  rows_valid = {len(df_valid)}")
    print(f"  elapsed    = {elapsed:.2f} s")

    if fit is None:
        print("  fit        = None")
    elif "error" in fit:
        print(f"  fit ERROR  = {fit['error']}")
    else:
        print(f"  a          = {fit.get('a'):.6f}")
        print(f"  b          = {fit.get('b'):.6f}")
        print(f"  c          = {fit.get('c'):.6f}")
        print(f"  R2_raw     = {fit.get('R2_raw_beta'):.6f}")
        print(f"  MAE unc    = {fit.get('MAE_uncorrected_eta_pct'):.2f}%")
        print(f"  MAE corr   = {fit.get('MAE_corrected_eta_pct'):.2f}%")
        print(f"  improve    = {fit.get('improvement_factor'):.2f}x")
        print(f"  n_used     = {fit.get('n_points_used')}")
        print(f"  n_total    = {fit.get('n_points_total')}")

    print("\nWROTE")
    print(f"  {p_all}")
    print(f"  {p_valid}")
    print(f"  {p_fit}")
    print(f"  {p_summary}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
