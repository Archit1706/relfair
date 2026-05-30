"""
Tests for relfair.counterfactual.

The key invariant being tested is the difference between naive and
relationship-aware counterfactual generation — this is the research claim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relfair.counterfactual import (
    CounterfactualRecord,
    batch_counterfactuals,
    flip_rate,
    generate_counterfactual,
)
from relfair.mechanisms import LearnedMechanisms


@pytest.fixture
def fitted_mechs(simple_graph, toy_df):
    mechs = LearnedMechanisms(simple_graph)
    mechs.fit(toy_df)
    return mechs


@pytest.fixture
def male_row():
    return {"sex": "M", "honorific": "Mr", "marital": "Married", "income_k": 60.0}


# ---------------------------------------------------------------------------
# Basic generation
# ---------------------------------------------------------------------------

def test_generate_returns_record(simple_graph, fitted_mechs, male_row):
    rec = generate_counterfactual(male_row, "sex", "F", simple_graph, fitted_mechs, seed=0)
    assert isinstance(rec, CounterfactualRecord)
    assert rec.original["sex"] == "M"
    assert rec.counterfactual["sex"] == "F"


def test_original_row_is_not_mutated(simple_graph, fitted_mechs, male_row):
    original_copy = dict(male_row)
    generate_counterfactual(male_row, "sex", "F", simple_graph, fitted_mechs, seed=0)
    assert male_row == original_copy


# ---------------------------------------------------------------------------
# The core distinction: naive vs. relationship-aware
# ---------------------------------------------------------------------------

def test_naive_leaves_honorific_unchanged(simple_graph, fitted_mechs, male_row):
    """Naive flip: sex=M→F but honorific stays Mr — the incoherence we're fixing."""
    rec = generate_counterfactual(
        male_row, "sex", "F", simple_graph, fitted_mechs, naive=True, seed=0
    )
    assert rec.counterfactual["honorific"] == "Mr"   # incoherent — left unchanged
    assert rec.coflips == {}                          # no co-flips recorded


def test_relaware_fixes_honorific(simple_graph, fitted_mechs, male_row):
    """Relationship-aware flip: sex=M→F triggers honorific Mr→Ms via hard rule."""
    rec = generate_counterfactual(
        male_row, "sex", "F", simple_graph, fitted_mechs, naive=False, seed=0
    )
    assert rec.counterfactual["honorific"] == "Ms"
    assert "honorific" in rec.coflips
    assert rec.coflips["honorific"]["from"] == "Mr"
    assert rec.coflips["honorific"]["to"] == "Ms"


def test_relaware_propagates_all_descendants(simple_graph, fitted_mechs, male_row):
    """All three descendants of sex must appear in the counterfactual (possibly changed)."""
    rec = generate_counterfactual(
        male_row, "sex", "F", simple_graph, fitted_mechs, naive=False, seed=0
    )
    cf = rec.counterfactual
    assert "honorific" in cf
    assert "marital" in cf
    assert "income_k" in cf


def test_non_descendant_cols_unchanged(simple_graph, fitted_mechs):
    """Columns not in the graph are passed through unchanged."""
    row = {"sex": "M", "honorific": "Mr", "marital": "Married", "income_k": 60.0,
           "extra_col": "should_not_change"}
    rec = generate_counterfactual(row, "sex", "F", simple_graph, fitted_mechs, naive=False, seed=0)
    assert rec.counterfactual["extra_col"] == "should_not_change"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_same_seed_produces_same_counterfactual(simple_graph, fitted_mechs, male_row):
    rec1 = generate_counterfactual(male_row, "sex", "F", simple_graph, fitted_mechs, seed=42)
    rec2 = generate_counterfactual(male_row, "sex", "F", simple_graph, fitted_mechs, seed=42)
    assert rec1.counterfactual == rec2.counterfactual


def test_different_seeds_may_differ_for_learned_nodes(simple_graph, fitted_mechs, male_row):
    """income_k has a learned mechanism with noise — different seeds should differ."""
    results = set()
    for s in range(20):
        rec = generate_counterfactual(
            male_row, "sex", "F", simple_graph, fitted_mechs, naive=False, seed=s
        )
        results.add(round(rec.counterfactual["income_k"], 4))
    assert len(results) > 1


# ---------------------------------------------------------------------------
# Batch + flip rate
# ---------------------------------------------------------------------------

def test_batch_length_matches_dataframe(simple_graph, fitted_mechs, toy_df):
    records = batch_counterfactuals(toy_df, "sex", "F", simple_graph, fitted_mechs, seed=0)
    assert len(records) == len(toy_df)


def test_flip_rate_naive_vs_relaware(simple_graph, fitted_mechs, toy_df):
    """
    Core research claim: the naive flip rate on honorific is higher than
    the relationship-aware rate, because naive leaves Husband/Mr intact
    on rows where sex was already M (creating incoherent inputs).

    Here we use honorific as a synthetic 'model' output to make this
    concrete and deterministic.
    """
    # subset: Male rows where honorific=Mr (the incoherent case)
    male_rows = toy_df[toy_df["sex"] == "M"].copy()

    naive_recs = batch_counterfactuals(
        male_rows, "sex", "F", simple_graph, fitted_mechs, naive=True, seed=0
    )
    relaware_recs = batch_counterfactuals(
        male_rows, "sex", "F", simple_graph, fitted_mechs, naive=False, seed=0
    )

    # "model": predicts 1 if honorific == "Mr", else 0
    # Naive CF leaves honorific=Mr → prediction 1 → no flip (false negative)
    # But for detecting discrimination the right signal is: honorific changed
    # Use honorific as a proxy target: flip = honorific changed
    def honorific_pred(df: pd.DataFrame):
        return (df["honorific"] == "Mr").astype(int).values

    naive_result = flip_rate(naive_recs, honorific_pred)
    relaware_result = flip_rate(relaware_recs, honorific_pred)

    # Naive: honorific is never flipped (left as-is)
    assert naive_result["flip_rate"] == pytest.approx(0.0)
    # Relationship-aware: honorific is always flipped (hard rule)
    assert relaware_result["flip_rate"] == pytest.approx(1.0)


def test_flip_rate_with_on_manifold_mask(simple_graph, fitted_mechs, toy_df):
    records = batch_counterfactuals(toy_df, "sex", "F", simple_graph, fitted_mechs, seed=0)

    def constant_predict(df):
        return np.zeros(len(df), dtype=int)

    mask = [True] * (len(records) // 2) + [False] * (len(records) - len(records) // 2)
    result = flip_rate(records, constant_predict, on_manifold_mask=mask)
    assert result["total"] == len(records) // 2


def test_detect_constraint_violations_catches_husband_female(simple_graph, fitted_mechs):
    """
    The canonical false-positive case: a naive Male->Female CF on a Husband row
    leaves relationship=Husband, violating the hard rule that says Female->Wife.
    detect_constraint_violations must flag that CF.

    Separate from Own-child which has its own passthrough rule.
    """
    from relfair.counterfactual import detect_constraint_violations
    from relfair.graph import DependencyGraph

    # Graph with ONLY the relationship dependency (no honorific, to isolate the test)
    G = DependencyGraph.from_edges([("sex", "relationship")])
    G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife",      from_val="Husband")
    G.add_hard_rule("relationship", when={"sex": "Female"}, value="Own-child", from_val="Own-child")

    husband_row = {"sex": "Male", "relationship": "Husband"}
    child_row   = {"sex": "Male", "relationship": "Own-child"}

    # Naive CFs — relationship stays unchanged by definition
    naive_husband = generate_counterfactual(husband_row, "sex", "Female", G, fitted_mechs, naive=True)
    naive_child   = generate_counterfactual(child_row,   "sex", "Female", G, fitted_mechs, naive=True)

    violated = detect_constraint_violations([naive_husband, naive_child], G)

    # Husband: CF = {sex=Female, relationship=Husband} violates Husband->Wife rule
    assert bool(violated[0]), "Husband naive CF must be flagged as constraint-violated"
    # Own-child: CF = {sex=Female, relationship=Own-child} satisfies its passthrough rule
    assert not violated[1], "Own-child naive CF must NOT be flagged (passthrough satisfied)"


def test_detect_constraint_violations_relaware_clean(simple_graph, fitted_mechs):
    """Rel-aware CFs should never be constraint-violated — the whole point."""
    from relfair.counterfactual import detect_constraint_violations
    from relfair.graph import DependencyGraph

    G = DependencyGraph.from_edges([("sex", "relationship"), ("sex", "honorific")])
    G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife",     from_val="Husband")
    G.add_hard_rule("relationship", when={"sex": "Female"}, value="Own-child", from_val="Own-child")
    G.add_hard_rule("honorific",    when={"sex": "Female"}, value="Ms")

    husband_row = {"sex": "Male", "relationship": "Husband", "honorific": "Mr"}
    rel_aware   = generate_counterfactual(husband_row, "sex", "Female", G, fitted_mechs, naive=False, seed=0)
    violated    = detect_constraint_violations([rel_aware], G)
    assert not violated[0], "Rel-aware CF must not violate constraints"


def test_flip_rate_empty_mask():
    """All False mask → flip_rate is 0 and total is 0."""
    from relfair.counterfactual import flip_rate
    # No records passed (empty on-manifold mask equivalent)
    result = flip_rate([], lambda df: np.array([]), on_manifold_mask=[])
    assert result["total"] == 0
    assert result["flip_rate"] == pytest.approx(0.0)
