"""
Deterministic tests for paper grids. Do NOT depend on Monte Carlo.
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.slow

from buffer_policy.experiments_paper import (
    PAPER_P_GRID,
    PAPER_SMAX_CONFIGS,
    PAPER_SMAX_VALUES,
    PAPER_UTIL_GRID,
    run_baseline_paper,
    run_hazard_paper,
    run_heatmap_paper,
    run_lemma1_audit_paper,
    run_misspecification_paper,
    run_smax_paper,
)
from buffer_policy.kernel import build_kernel
from buffer_policy.mdp import optimize_threshold, run_rvi
from buffer_policy.params import CostParams, ModelParams


def test_paper_grids_have_expected_lengths() -> None:
    assert len(PAPER_P_GRID) == 6
    assert len(PAPER_UTIL_GRID) == 6
    assert len(PAPER_SMAX_VALUES) == 6
    assert len(PAPER_SMAX_CONFIGS) == 3


def test_paper_baseline_s_star() -> None:
    r = run_baseline_paper()
    assert r["s_star"] == 38, f"baseline s_star expected 38, got {r['s_star']}"


def test_paper_baseline_rvi_match_threshold() -> None:
    """Threshold optimization and RVI should agree on the average cost."""
    mp = ModelParams(p=0.15, lam=80.0, C=100, S_max=180, X0=50, s_reset=50)
    cp = CostParams(K_f=100.0, K_p=10.0)
    P = build_kernel(mp)
    s_star, g_star, _ = optimize_threshold(mp, cp, P=P)
    rvi = run_rvi(mp, cp, P=P)
    assert s_star == 38
    assert abs(g_star - rvi.g) / rvi.g < 1e-6


def test_paper_heatmap_cardinality() -> None:
    df = run_heatmap_paper()
    assert len(df) == 36, f"heatmap should have 36 rows, got {len(df)}"
    assert int(df["valid"].sum()) == 21, \
        f"expected 21 valid, got {int(df['valid'].sum())}"


def test_paper_hazard_cardinality() -> None:
    df = run_hazard_paper()
    assert len(df) == 36
    assert int(df["valid"].sum()) == 21


def test_paper_smax_cardinality() -> None:
    df = run_smax_paper()
    assert len(df) == 18, f"smax should have 18 rows, got {len(df)}"


def test_paper_lemma1_holds_all_valid_configs() -> None:
    df = run_lemma1_audit_paper()
    valid = df[df["valid"]]
    assert len(valid) == 21
    assert valid["lemma1_holds"].all(), \
        "Lemma 1 should hold for every valid config"


def test_paper_misspecification_zero_regret_at_truth() -> None:
    """At err_pct=0, regret should be exactly 0 (using true model)."""
    df = run_misspecification_paper()
    sub = df[(df["param_err"] == "p")
             & (df["err_pct"] == 0)
             & (df["valid"])]
    assert len(sub) == 1, \
        f"expected one row with p err=0, got {len(sub)}"
    assert abs(sub["regret_pct"].iloc[0]) < 1e-9, \
        f"regret at truth should be 0, got {sub['regret_pct'].iloc[0]}"


def test_paper_misspecification_lambda_zero_regret_at_truth() -> None:
    df = run_misspecification_paper()
    sub = df[(df["param_err"] == "lam")
             & (df["err_pct"] == 0)
             & (df["valid"])]
    assert len(sub) == 1
    assert abs(sub["regret_pct"].iloc[0]) < 1e-9


def test_paper_heatmap_g_CBM_le_g_RTF() -> None:
    df = run_heatmap_paper()
    valid = df[df["valid"]]
    assert (valid["g_CBM"] <= valid["g_RTF"] + 1e-9).all()


def test_paper_smax_monotone_E_tau_within_config() -> None:
    """E_tau should grow with S_max for fixed (p, util)."""
    df = run_smax_paper()
    for cfg_name, sub in df[df["valid"]].groupby("config"):
        sub_sorted = sub.sort_values("S_max")
        e = sub_sorted["E_tau"].to_numpy()
        diffs = np.diff(e)
        assert (diffs >= -1e-6).all(), \
            f"E_tau not monotone in {cfg_name}: {e}"
