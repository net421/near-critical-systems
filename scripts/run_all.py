#!/usr/bin/env python
"""
End-to-end driver: chains Phase 1 + Phase 2 + Phase 3 + Phase 4
as subprocesses so each phase failure isolates cleanly.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def banner(s: str) -> None:
    print("\n" + "#" * 72)
    print("# " + s)
    print("#" * 72)


def run_step(name: str, cmd: list[str]) -> int:
    banner(f"STEP: {name}")
    print(f"  cmd: {' '.join(cmd)}")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    print(f"  exit code: {rc}")
    return rc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run all phases")
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    ap.add_argument("--skip-beta", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out", type=str, default="results/phase_4")
    ap.add_argument("--seed", type=int, default=20260429)
    ap.add_argument("--skip-phase-1", action="store_true")
    ap.add_argument("--skip-phase-2", action="store_true")
    ap.add_argument("--skip-phase-3", action="store_true")
    ap.add_argument("--skip-phase-4", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    py = sys.executable
    failures: list[str] = []

    if not args.skip_phase_1:
        script = ROOT / "scripts" / "run_phase_1.py"
        if script.exists():
            rc = run_step("Phase 1", [py, str(script)])
            if rc != 0:
                failures.append("phase_1")
        else:
            print(f"[run_all] WARNING: {script} not found; skipping phase 1")

    if not args.skip_phase_2:
        script = ROOT / "scripts" / "run_phase_2.py"
        if script.exists():
            rc = run_step("Phase 2", [py, str(script)])
            if rc != 0:
                failures.append("phase_2")
        else:
            print(f"[run_all] WARNING: {script} not found; skipping phase 2")

    if not args.skip_phase_3:
        script = ROOT / "scripts" / "run_phase_3.py"
        if script.exists():
            rc = run_step("Phase 3", [py, str(script)])
            if rc != 0:
                failures.append("phase_3")
        else:
            print(f"[run_all] WARNING: {script} not found; skipping phase 3")

    if not args.skip_phase_4:
        script = ROOT / "scripts" / "run_phase_4.py"
        cmd = [
            py, str(script),
            "--mode", args.mode,
            "--out", args.out,
            "--seed", str(args.seed),
        ]
        if args.skip_beta:
            cmd.append("--skip-beta")
        if args.no_plots:
            cmd.append("--no-plots")
        rc = run_step("Phase 4", cmd)
        if rc != 0:
            failures.append("phase_4")

    banner("ALL PHASES COMPLETE")
    if failures:
        print(f"  FAILURES: {', '.join(failures)}")
        return 1
    print("  ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
