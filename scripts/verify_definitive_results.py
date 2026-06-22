#!/usr/bin/env python
"""Verify the archived numerical results used by the manuscript."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load_json(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print("PASS:", message)


def close(actual: float, expected: float, tolerance: float = 1e-9) -> bool:
    return math.isclose(float(actual), expected, rel_tol=tolerance,
                        abs_tol=tolerance)


def main() -> int:
    beta_dir = RESULTS / "beta_calibration"
    summary = load_json(beta_dir / "summary_n15000_T300000.json")
    fit = summary["fit"]
    points = load_csv(beta_dir / "beta_points_n15000_T300000.csv")

    require(summary["config"]["n_runs"] == 15_000,
            "15,000 replications per design point")
    require(summary["config"]["max_steps"] == 300_000,
            "300,000-period simulation horizon")
    require(summary["config"]["seed"] == 20_260_429,
            "master seed 20260429")
    require(summary["config"]["seed_mode"] == "SeedSequence.spawn",
            "corrected independent-stream seed mode")
    require(len(points) == 42 and fit["n_points_used"] == 42,
            "42 fitted calibration points")

    expected = {
        "a": 0.24771504913644563,
        "b": 0.3929272835920071,
        "c": 0.011126659571810244,
        "R2_raw_beta": 0.9535754625284067,
        "MAE_uncorrected_eta_pct": 36.01261645502646,
        "MAE_corrected_eta_pct": 5.010951745494996,
        "max_error_uncorrected_eta_pct": 62.6378426849007,
        "max_error_corrected_eta_pct": 18.382955387682184,
        "improvement_factor": 7.186781730118025,
    }
    for key, value in expected.items():
        require(close(fit[key], value), f"{key} = {value:.10g}")

    paper = RESULTS / "paper_tables"
    hazard = load_json(paper / "diagnostic_hazard_summary_paper.json")
    require(hazard["total_configs"] == 21,
            "21 feasible hazard configurations")
    require(hazard["DFR_approximate_total"] == 21,
            "approximate DFR in all feasible configurations")
    require(hazard["DFR_strict_total"] == 11,
            "strict smoothed DFR in 11 configurations")

    cost_rows = load_csv(paper / "exp_cost_ratio_paper.csv")
    require(len(cost_rows) == 5, "five cost-ratio cases")
    max_gain = max(float(row["gain_vs_AB_best_pct"]) for row in cost_rows)
    require(close(max_gain, 21.5582107707877),
            "maximum gain over age-based benchmark is 21.5582%")

    heatmap = load_csv(paper / "exp_heatmap_paper.csv")
    require(sum(row["valid"].lower() == "true" for row in heatmap) == 21,
            "21 valid operating-grid cases")

    misspec = load_csv(paper / "exp_misspecification_paper.csv")
    valid_regret = [float(row["regret_pct"]) for row in misspec
                    if row["valid"].lower() == "true"]
    require(max(valid_regret) < 6.0,
            "maximum valid misspecification regret is below 6%")

    lemma = load_csv(paper / "lemma1_audit_paper.csv")
    valid_lemma = [row for row in lemma if row["valid"].lower() == "true"]
    require(len(valid_lemma) == 21 and all(
        row["lemma1_holds"].lower() == "true" for row in valid_lemma
    ), "conditional lemma holds in all 21 valid cases")

    print("\nAll definitive manuscript results verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
