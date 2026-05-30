"""
Tests for relfair.metrics — LL 144 aggregate metrics engine.

Uses the Northbound Talent mock data from design/data.js as the reference
dataset (real-shaped selection rates and intersectional cells).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from relfair.metrics import (
    FOUR_FIFTHS_THRESHOLD,
    SMALL_SAMPLE_THRESHOLD,
    GroupStat,
    LL144Result,
    compute_ll144_metrics,
    fisher_exact_pvalue,
    bootstrap_ci,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_df() -> pd.DataFrame:
    """
    Synthetic hiring dataset mirroring Northbound Talent mock data.
    Male selection rate ~26.4%, Female ~23.1%, Non-binary ~18.8%.
    White rate ~30.8%, Black ~20.7%, Native Hawaiian ~19%.
    """
    rng = np.random.default_rng(42)
    n   = 2000

    sex_vals  = rng.choice(["Male", "Female", "Non-binary"], size=n, p=[0.51, 0.47, 0.02])
    race_vals = rng.choice(
        ["White", "Asian", "Hispanic or Latino", "Black or African American",
         "Two or More Races", "Native Hawaiian or Pacific Islander",
         "American Indian or Alaska Native"],
        size=n, p=[0.37, 0.22, 0.17, 0.14, 0.04, 0.03, 0.03],
    )

    # Group-specific selection probabilities
    sex_rates  = {"Male": 0.264, "Female": 0.231, "Non-binary": 0.188}
    race_rates = {
        "White": 0.308, "Asian": 0.289, "Two or More Races": 0.264,
        "Hispanic or Latino": 0.247, "Black or African American": 0.207,
        "Native Hawaiian or Pacific Islander": 0.190,
        "American Indian or Alaska Native": 0.159,
    }

    probs = np.array([
        (sex_rates[sx] + race_rates[rc]) / 2
        for sx, rc in zip(sex_vals, race_vals)
    ])
    hired = (rng.random(n) < probs).astype(int)

    return pd.DataFrame({"sex": sex_vals, "race": race_vals, "hired": hired})


@pytest.fixture
def score_df() -> pd.DataFrame:
    """Scoring AEDT: score column + same demographics."""
    rng = np.random.default_rng(7)
    n   = 1000
    sex  = rng.choice(["Male", "Female"], size=n, p=[0.55, 0.45])
    race = rng.choice(["White", "Black or African American", "Asian"], size=n, p=[0.6, 0.2, 0.2])
    score = np.where(sex == "Male", rng.normal(72, 15, n), rng.normal(65, 15, n)).clip(0, 100)
    return pd.DataFrame({"sex": sex, "race": race, "score": score})


@pytest.fixture
def result(binary_df) -> LL144Result:
    return compute_ll144_metrics(
        binary_df,
        outcome_col="hired",
        outcome_type="binary",
        sex_col="sex",
        race_col="race",
        n_boot=200,   # fast for tests; use 2000 in production
        bootstrap_seed=0,
        meta={"employer": "Acme Corp", "aedt_name": "Ranker v1"},
    )


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------

def test_result_type(result):
    assert isinstance(result, LL144Result)


def test_totals_add_up(binary_df, result):
    assert result.n_total == len(binary_df)
    assert result.n_selected == int(binary_df["hired"].sum())
    assert abs(result.overall_rate - binary_df["hired"].mean()) < 1e-9


def test_by_sex_all_groups_present(binary_df, result):
    expected = set(binary_df["sex"].unique())
    actual   = {s.group for s in result.by_sex}
    assert expected == actual


def test_by_race_all_groups_present(binary_df, result):
    expected = set(binary_df["race"].unique())
    actual   = {s.group for s in result.by_race}
    assert expected == actual


def test_reference_group_has_ratio_one(result):
    ref = next(s for s in result.by_sex if s.is_reference)
    assert abs(ref.ratio - 1.0) < 1e-9


def test_all_ratios_leq_one(result):
    for s in result.by_sex + result.by_race:
        assert s.ratio <= 1.0 + 1e-9, f"{s.group}: ratio={s.ratio:.4f} > 1"


def test_group_with_lowest_rate_may_be_flagged(result):
    """The group with the lowest selection rate should have the lowest ratio.
    If it's below 0.80, it must be flagged; the flag logic is what we test,
    not a hard-coded expected outcome from a stochastic fixture."""
    all_groups = result.by_sex
    if not all_groups:
        return
    worst = min(all_groups, key=lambda s: s.ratio)
    # The flag must be consistent with the ratio regardless of which group is worst
    assert worst.four_fifths_flag == (worst.ratio < FOUR_FIFTHS_THRESHOLD)


def test_four_fifths_threshold(result):
    """All groups with ratio < 0.80 must be flagged, all >= 0.80 must not be."""
    for s in result.by_sex + result.by_race:
        if s.ratio < FOUR_FIFTHS_THRESHOLD:
            assert s.four_fifths_flag, f"{s.group}: ratio={s.ratio:.3f} < 0.80 not flagged"
        else:
            assert not s.four_fifths_flag, f"{s.group}: ratio={s.ratio:.3f} >= 0.80 flagged"


def test_ci_bounds_straddle_point_estimate(result):
    """For groups with sufficient n, the CI should contain the point estimate."""
    for s in result.by_sex + result.by_race:
        if not s.small_sample:
            assert s.ratio_ci_lo <= s.ratio + 1e-9, (
                f"{s.group}: ci_lo={s.ratio_ci_lo:.3f} > ratio={s.ratio:.3f}"
            )
            assert s.ratio_ci_hi >= s.ratio - 1e-9, (
                f"{s.group}: ci_hi={s.ratio_ci_hi:.3f} < ratio={s.ratio:.3f}"
            )


def test_reference_group_has_no_p_value(result):
    ref = next(s for s in result.by_sex if s.is_reference)
    assert ref.p_value is None


def test_non_reference_groups_have_p_values(result):
    for s in result.by_sex:
        if not s.is_reference:
            assert s.p_value is not None


def test_small_sample_flag(binary_df, result):
    """Non-binary group (n≈40) should be flagged as small sample."""
    nb = next((s for s in result.by_sex if s.group == "Non-binary"), None)
    if nb is not None:
        assert nb.small_sample == (nb.n < SMALL_SAMPLE_THRESHOLD)


# ---------------------------------------------------------------------------
# Intersectional tests
# ---------------------------------------------------------------------------

def test_intersectional_cell_count(binary_df, result):
    n_sex  = binary_df["sex"].nunique()
    n_race = binary_df["race"].nunique()
    assert len(result.intersectional) == n_sex * n_race


def test_intersectional_reference_ratio_one(result):
    max_ratio = max(c.ratio for c in result.intersectional)
    assert abs(max_ratio - 1.0) < 1e-9


def test_intersectional_flags(result):
    for cell in result.intersectional:
        expected = cell.ratio < FOUR_FIFTHS_THRESHOLD
        assert cell.four_fifths_flag == expected


# ---------------------------------------------------------------------------
# Outcome type: score
# ---------------------------------------------------------------------------

def test_score_outcome_type(score_df):
    r = compute_ll144_metrics(
        score_df,
        outcome_col="score",
        outcome_type="score",
        sex_col="sex",
        race_col="race",
        n_boot=100,
    )
    assert r.outcome_type == "score"
    assert r.threshold is not None
    assert abs(r.threshold - float(score_df["score"].median())) < 1e-9


def test_invalid_outcome_type(binary_df):
    with pytest.raises(ValueError, match="outcome_type"):
        compute_ll144_metrics(
            binary_df, outcome_col="hired",
            outcome_type="bad", sex_col="sex", race_col="race",
        )


# ---------------------------------------------------------------------------
# Statistical utilities
# ---------------------------------------------------------------------------

def test_bootstrap_ci_mean():
    rng = np.random.default_rng(0)
    data = pd.DataFrame({"x": rng.normal(0.5, 0.1, 1000)})
    lo, hi = bootstrap_ci(data, lambda d: float(d["x"].mean()), n_boot=1000, seed=0)
    assert lo < 0.5 < hi
    assert hi - lo < 0.05   # tight CI for n=1000


def test_fisher_exact_detects_disparity():
    """Fisher's exact must flag a strong, unambiguous disparity."""
    # 200 White: 100 selected (50%); 200 Black: 60 selected (30%) → clear gap
    df = pd.DataFrame({
        "race":  ["White"] * 200 + ["Black or African American"] * 200,
        "hired": [1] * 100 + [0] * 100 + [1] * 60 + [0] * 140,
    })
    p = fisher_exact_pvalue(df, "race", "hired", "White", "Black or African American")
    assert p is not None
    assert p < 0.001, f"Expected highly significant disparity, got p={p:.6f}"


def test_fisher_exact_empty_group(binary_df):
    p = fisher_exact_pvalue(binary_df, "sex", "hired", "Male", "Nonexistent")
    assert p is None


# ---------------------------------------------------------------------------
# Meta passthrough
# ---------------------------------------------------------------------------

def test_meta_passthrough(binary_df):
    r = compute_ll144_metrics(
        binary_df, outcome_col="hired", outcome_type="binary",
        sex_col="sex", race_col="race", n_boot=50,
        meta={"employer": "Test Corp", "aedt_name": "Model v1", "audit_date": "2026-05-21"},
    )
    assert r.meta["employer"] == "Test Corp"
    assert r.meta["aedt_name"] == "Model v1"
