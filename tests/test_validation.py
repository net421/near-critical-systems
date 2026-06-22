"""Tests for validation.py: Lemma-1 and threshold-vs-RVI audits."""
from __future__ import annotations

from buffer_policy.params import baseline_cost, baseline_model
from buffer_policy.validation import audit_lemma1, audit_threshold_vs_rvi


def test_lemma1_baseline() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    res = audit_lemma1(mp, cp)
    assert res.premise_holds
    assert res.holds
    assert res.s_star_wide < mp.s_reset
    assert res.g_star_wide < cp.K_p


def test_threshold_vs_rvi_baseline() -> None:
    mp = baseline_model()
    cp = baseline_cost()
    res = audit_threshold_vs_rvi(mp, cp)
    assert res.rvi_is_threshold
    assert res.rvi_s_star is not None
    assert res.cost_rel_diff < 1e-6
