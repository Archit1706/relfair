"""Tests for relfair.mechanisms — LearnedMechanisms."""

import numpy as np
import pandas as pd
import pytest
from relfair.mechanisms import LearnedMechanisms


@pytest.fixture
def fitted_mechs(simple_graph, toy_df):
    mechs = LearnedMechanisms(simple_graph)
    mechs.fit(toy_df)
    return mechs


def test_fit_produces_mechanisms_for_non_roots(fitted_mechs):
    fitted = set(fitted_mechs.fitted_nodes())
    # sex is a root — no mechanism
    assert "sex" not in fitted
    # descendants of sex get mechanisms
    assert {"honorific", "marital", "income_k"}.issubset(fitted)


def test_sample_categorical_returns_valid_value(fitted_mechs):
    rng = np.random.default_rng(42)
    val = fitted_mechs.sample("honorific", {"sex": "M"}, rng)
    assert val in ("Mr", "Ms")


def test_sample_continuous_returns_float(fitted_mechs):
    rng = np.random.default_rng(42)
    val = fitted_mechs.sample("income_k", {"sex": "M"}, rng)
    assert isinstance(val, float)
    assert 10 < val < 120   # sanity range


def test_sample_unknown_node_raises(fitted_mechs):
    rng = np.random.default_rng(0)
    with pytest.raises(LookupError):
        fitted_mechs.sample("nonexistent", {"sex": "M"}, rng)


def test_sample_reproducible_with_same_seed(fitted_mechs):
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    v1 = fitted_mechs.sample("income_k", {"sex": "F"}, rng1)
    v2 = fitted_mechs.sample("income_k", {"sex": "F"}, rng2)
    assert v1 == v2


def test_sample_batch_matches_sample_distribution(fitted_mechs):
    """sample_batch and sample() should draw from the same distribution."""
    rng_b = np.random.default_rng(0)
    rng_s = np.random.default_rng(0)
    parent_df = pd.DataFrame([{"sex": "F"}] * 100)

    batch_vals  = fitted_mechs.sample_batch("income_k", parent_df, rng_b)
    single_vals = [fitted_mechs.sample("income_k", {"sex": "F"}, rng_s) for _ in range(100)]

    # Same RNG seed → same values
    assert batch_vals == single_vals


def test_sample_batch_returns_correct_length(fitted_mechs, toy_df):
    rng = np.random.default_rng(42)
    parent_df = toy_df[["sex"]].rename(columns={"sex": "sex"}).iloc[:50]
    vals = fitted_mechs.sample_batch("honorific", parent_df, rng)
    assert len(vals) == 50
    assert all(v in ("Mr", "Ms") for v in vals)


def test_mechanism_learns_sex_income_direction(fitted_mechs):
    """Male income predictions should on average be higher than Female (data-driven)."""
    male_samples = [fitted_mechs.sample("income_k", {"sex": "M"}, np.random.default_rng(i)) for i in range(50)]
    female_samples = [fitted_mechs.sample("income_k", {"sex": "F"}, np.random.default_rng(1000 + i)) for i in range(50)]
    assert np.mean(male_samples) > np.mean(female_samples)
