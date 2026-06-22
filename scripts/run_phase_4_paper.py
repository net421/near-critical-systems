#!/usr/bin/env python
"""
Paper-grade Phase 4 validator.

Steps:
  1. Run run_all_paper.py --beta-quick --no-plots
  2. Verify CSV outputs exist and are non-empty
  3. Verify final_summary_paper.json keys
  4. Run compare_to_paper.py
  5. Check mathematical invariants
  6. Run pytest

Exit code 0 if all checks PASS, 1 otherwise.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def banner(s: str) -> None:
    print("\n" + "#" * 72)
    print("# [phase4_paper] " + s)
    print("#" * 72)


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")


def main() -> int:
    out_root = ROOT / "results" / "full_paper"
    paper_dir = out_root / "paper"
    py = sys.executable
    failures: list[str] = []

    # 1. run_all_paper.py
    banner("step 1: run_all_paper.py --beta-quick --no-plots")
    rc = subprocess.call(
        [py, str(ROOT / "scripts" / "run_all_paper.py"),
         "--beta-quick", "--no-plots",
         "--out", str(out_root)],
        cwd=str(ROOT),
    )
    if rc != 0:
        failures.append(f"run_all_paper exited with code {rc}")
        print(f"  exit code: {rc}")
    else:
        print("  PASS")

    # 2. CSV outputs exist and non-empty
    banner("step 2: checking CSV outputs")
    required = [
        "exp_cost_ratio_paper.csv",
        "exp_heatmap_paper.csv",
        "exp_smax_paper.csv",
        "exp_smax_summary_paper.csv",
        "exp_misspecification_paper.csv",
        "diagnostic_hazard_paper.csv",
        "lemma1_audit_paper.csv",
        "exp_beta_points_paper.csv",
        "exp_beta_validity_paper.csv",
    ]
    for name in required:
        path = paper_dir / name
        if not path.exists():
            failures.append(f"missing CSV: {name}")
            fail(f"missing {name}")
        elif path.stat().st_size == 0:
            failures.append(f"empty CSV: {name}")
            fail(f"empty {name}")
        else:
            print(f"  OK {name} ({path.stat().st_size} bytes)")

    # 3. final_summary_paper.json keys
    banner("step 3: checking final_summary_paper.json")
    summary_path = out_root / "final_summary_paper.json"
    if not summary_path.exists():
        failures.append("final_summary_paper.json missing")
        fail("missing")
    else:
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            required_keys = ["metadata", "mode", "config",
                             "baseline", "experiments"]
            for k in required_keys:
                if k not in summary:
                    failures.append(f"summary missing key '{k}'")
                    fail(f"missing key {k}")
                else:
                    print(f"  OK has key '{k}'")
        except Exception as exc:
            failures.append(f"summary parse error: {exc}")
            fail(f"parse error: {exc}")

    # 4. compare_to_paper.py
    banner("step 4: compare_to_paper.py")
    cmp_script = ROOT / "scripts" / "compare_to_paper.py"
    if cmp_script.exists():
        rc = subprocess.call(
            [py, str(cmp_script), "--csv-dir", str(paper_dir),
             "--summary", str(summary_path)],
            cwd=str(ROOT),
        )
        if rc != 0:
            failures.append(f"compare_to_paper exit code {rc}")
        else:
            print("  PASS")
    else:
        print("  WARNING: compare_to_paper.py not found; skipping")

    # 5. mathematical invariants
    banner("step 5: checking mathematical invariants")
    try:
        df_cost = pd.read_csv(paper_dir / "exp_cost_ratio_paper.csv")
        viol = (df_cost["g_AB_best"] - df_cost["g_RTF"]) > 1e-9
        if viol.any():
            failures.append("g_AB_best > g_RTF")
            fail(f"  g_AB_best > g_RTF in {int(viol.sum())} rows")
        else:
            print("  OK g_AB_best <= g_RTF")

        viol = (df_cost["g_CBM"] - df_cost["g_RTF"]) > 1e-9
        if viol.any():
            failures.append("g_CBM > g_RTF")
            fail(f"  g_CBM > g_RTF in {int(viol.sum())} rows")
        else:
            print("  OK g_CBM <= g_RTF")

        df_smax = pd.read_csv(paper_dir / "exp_smax_paper.csv")
        df_smax_v = df_smax[df_smax["valid"]]
        viol = df_smax_v["S_max"] < df_smax_v["s_reset"]
        if viol.any():
            failures.append("S_max < s_reset")
            fail(f"  S_max < s_reset in {int(viol.sum())} rows")
        else:
            print("  OK S_max >= s_reset")

        df_mis = pd.read_csv(paper_dir / "exp_misspecification_paper.csv")
        df_mis_v = df_mis[df_mis["valid"]]
        viol = df_mis_v["regret_pct"] < -1e-7
        if viol.any():
            failures.append("regret_pct < 0")
            fail(f"  regret_pct < 0 in {int(viol.sum())} rows")
        else:
            print("  OK regret_pct >= 0")

    except Exception as exc:
        failures.append(f"invariants check error: {exc}")
        fail(f"  exception: {type(exc).__name__}: {exc}")

    # 6. pytest
    banner("step 6: pytest")
    rc = subprocess.call(
        [py, "-m", "pytest", "tests/", "-q"],
        cwd=str(ROOT),
    )
    if rc != 0:
        failures.append(f"pytest exit code {rc}")
    else:
        print("  PASS")

    # final
    banner("FINAL")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("  [phase4_paper] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
