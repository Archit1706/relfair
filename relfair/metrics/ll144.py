"""
NYC Local Law 144 aggregate metrics engine.

Computes selection/scoring rates, impact ratios, the four-fifths rule, and
intersectional sex × race cross-tabs exactly as defined in:
  - NYC Admin Code §20-871 and the DCWP Final Rule (Dec 2022)
  - DCWP guidance on intersectional categories (§5-303(b), effective 2024)

The output is a single ``LL144Result`` dataclass that drives both the
human-readable PDF report and the machine-readable JSON artifact.

Two outcome types
-----------------
``binary``
    AEDT makes a binary selection decision.
    selection_rate(g) = selected(g) / n(g)

``score``
    AEDT produces a numeric score.
    scoring_rate(g) = n(g scoring above sample median) / n(g)

Both compute impact ratios the same way:
    impact_ratio(g) = rate(g) / max_h rate(h)
    four_fifths_flag = impact_ratio(g) < 0.80
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .stats import (
    FOUR_FIFTHS_THRESHOLD,
    SMALL_SAMPLE_THRESHOLD,
    bootstrap_impact_ratio_cis,
    fisher_exact_pvalue,
)

# ---------------------------------------------------------------------------
# LL 144 required race/ethnicity categories (EEOC set)
# ---------------------------------------------------------------------------
EEOC_RACE_CATEGORIES = [
    "White",
    "Black or African American",
    "Hispanic or Latino",
    "Asian",
    "Native Hawaiian or Pacific Islander",
    "American Indian or Alaska Native",
    "Two or More Races",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GroupStat:
    """Metrics for one demographic group within one protected attribute."""
    group:            str
    n:                int      # total applicants / scored individuals
    selected:         int      # n selected or scoring above threshold
    rate:             float    # selection / scoring rate
    ratio:            float    # impact ratio vs. reference group
    ratio_ci_lo:      float    # 95% bootstrap CI lower bound
    ratio_ci_hi:      float    # 95% bootstrap CI upper bound
    four_fifths_flag: bool     # ratio < 0.80
    p_value:          float | None  # Fisher's exact vs. reference (None = reference)
    small_sample:     bool     # n < SMALL_SAMPLE_THRESHOLD
    is_reference:     bool     # this group has the highest rate


@dataclass
class IntersectionalStat:
    """One cell of the sex × race intersectional table."""
    sex:              str
    race:             str
    n:                int
    selected:         int
    rate:             float
    ratio:            float    # vs. highest cell across all 14
    four_fifths_flag: bool
    small_sample:     bool


@dataclass
class LL144Result:
    """Complete result from one LL 144 audit computation."""
    # Aggregate
    n_total:      int
    n_selected:   int
    overall_rate: float
    outcome_type: str          # "binary" or "score"
    threshold:    float | None # median threshold when outcome_type="score"

    # Protected-attribute breakdowns
    by_sex:         list[GroupStat]
    by_race:        list[GroupStat]
    intersectional: list[IntersectionalStat]

    # Which groups are the reference (highest rate)
    reference_sex:  str
    reference_race: str

    # Arbitrary metadata (employer name, AEDT name, auditor, etc.)
    meta: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_ll144_metrics(
    df: pd.DataFrame,
    *,
    outcome_col: str,
    outcome_type: str = "binary",
    sex_col: str,
    race_col: str,
    n_boot: int = 2000,
    bootstrap_seed: int = 0,
    meta: dict[str, Any] | None = None,
) -> LL144Result:
    """
    Compute all LL 144 metrics from a predictions + demographics DataFrame.

    Args:
        df:               DataFrame with at least *outcome_col*, *sex_col*, *race_col*.
        outcome_col:      Column name of the binary outcome (0/1) or score.
        outcome_type:     "binary" (selection) or "score" (above-median rate).
        sex_col:          Column name for sex / gender.
        race_col:         Column name for race / ethnicity.
        n_boot:           Bootstrap resamples for impact-ratio CIs.
        bootstrap_seed:   RNG seed (pin for reproducibility).
        meta:             Arbitrary metadata forwarded to the result.

    Returns:
        LL144Result with all required metrics populated.
    """
    df = df.copy()

    # ── prepare the binary "selected" signal ──────────────────────────────
    if outcome_type == "binary":
        df["_selected"] = df[outcome_col].astype(int)
        threshold = None
    elif outcome_type == "score":
        threshold = float(df[outcome_col].median())
        df["_selected"] = (df[outcome_col] > threshold).astype(int)
    else:
        raise ValueError(f"outcome_type must be 'binary' or 'score', got {outcome_type!r}")

    n_total   = len(df)
    n_selected = int(df["_selected"].sum())
    overall_rate = n_selected / n_total if n_total > 0 else 0.0

    # ── by sex ────────────────────────────────────────────────────────────
    by_sex = _compute_group_stats(
        df, group_col=sex_col, selected_col="_selected",
        n_boot=n_boot, seed=bootstrap_seed,
    )
    ref_sex = _reference_group(by_sex)

    # ── by race ───────────────────────────────────────────────────────────
    by_race = _compute_group_stats(
        df, group_col=race_col, selected_col="_selected",
        n_boot=n_boot, seed=bootstrap_seed + 1,
    )
    ref_race = _reference_group(by_race)

    # ── intersectional (sex × race) ───────────────────────────────────────
    intersectional = _compute_intersectional(df, sex_col=sex_col, race_col=race_col)

    return LL144Result(
        n_total=n_total,
        n_selected=n_selected,
        overall_rate=overall_rate,
        outcome_type=outcome_type,
        threshold=threshold,
        by_sex=by_sex,
        by_race=by_race,
        intersectional=intersectional,
        reference_sex=ref_sex,
        reference_race=ref_race,
        meta=meta or {},
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_group_stats(
    df: pd.DataFrame,
    group_col: str,
    selected_col: str,
    n_boot: int,
    seed: int,
) -> list[GroupStat]:
    """Compute GroupStat for every unique value in *group_col*."""
    groups = sorted(df[group_col].dropna().unique(), key=str)
    if not groups:
        return []

    # Point estimates
    rates: dict[str, float] = {}
    for g in groups:
        sub = df[df[group_col] == g]
        rates[g] = float(sub[selected_col].mean()) if len(sub) > 0 else 0.0

    max_rate  = max(rates.values()) if rates else 1.0
    ref_group = max(rates, key=lambda g: rates[g])

    # Bootstrap CIs on impact ratios
    cis = bootstrap_impact_ratio_cis(df, group_col, selected_col, n_boot=n_boot, seed=seed)

    stats: list[GroupStat] = []
    for g in groups:
        sub       = df[df[group_col] == g]
        n_g       = len(sub)
        sel_g     = int(sub[selected_col].sum())
        rate_g    = rates[g]
        ratio_g   = rate_g / max_rate if max_rate > 0 else 0.0
        ci_lo, ci_hi = cis.get(g, (ratio_g, ratio_g))

        p_val = None if g == ref_group else fisher_exact_pvalue(
            df, group_col, selected_col, group_a=g, group_b=ref_group
        )

        stats.append(GroupStat(
            group=g,
            n=n_g,
            selected=sel_g,
            rate=rate_g,
            ratio=ratio_g,
            ratio_ci_lo=ci_lo,
            ratio_ci_hi=ci_hi,
            four_fifths_flag=ratio_g < FOUR_FIFTHS_THRESHOLD,
            p_value=p_val,
            small_sample=n_g < SMALL_SAMPLE_THRESHOLD,
            is_reference=(g == ref_group),
        ))

    # Sort: reference first, then descending by ratio
    stats.sort(key=lambda s: (-s.ratio, s.group))
    return stats


def _compute_intersectional(
    df: pd.DataFrame,
    sex_col: str,
    race_col: str,
) -> list[IntersectionalStat]:
    """
    LL 144 §5-303(b) intersectional sex × race cross-tab.

    Impact ratio for each cell = cell_rate / max(all_cell_rates).
    """
    cells: list[IntersectionalStat] = []
    sex_vals  = sorted(df[sex_col].dropna().unique(), key=str)
    race_vals = sorted(df[race_col].dropna().unique(), key=str)

    # Collect all (sex, race) group rates
    cell_rates: dict[tuple[str, str], tuple[int, int]] = {}
    for sx in sex_vals:
        for rc in race_vals:
            sub = df[(df[sex_col] == sx) & (df[race_col] == rc)]
            n_sub = len(sub)
            sel   = int(sub["_selected"].sum()) if n_sub > 0 else 0
            cell_rates[(sx, rc)] = (n_sub, sel)

    # Max rate across all cells
    rates = {
        k: (v[1] / v[0]) if v[0] > 0 else 0.0
        for k, v in cell_rates.items()
    }
    max_rate = max(rates.values()) if rates else 1.0

    for sx in sex_vals:
        for rc in race_vals:
            n_cell, sel_cell = cell_rates[(sx, rc)]
            rate_cell  = sel_cell / n_cell if n_cell > 0 else 0.0
            ratio_cell = rate_cell / max_rate if max_rate > 0 else 0.0
            cells.append(IntersectionalStat(
                sex=sx,
                race=rc,
                n=n_cell,
                selected=sel_cell,
                rate=rate_cell,
                ratio=ratio_cell,
                four_fifths_flag=ratio_cell < FOUR_FIFTHS_THRESHOLD,
                small_sample=n_cell < SMALL_SAMPLE_THRESHOLD,
            ))

    return cells


def _reference_group(stats: list[GroupStat]) -> str:
    for s in stats:
        if s.is_reference:
            return s.group
    return stats[0].group if stats else ""
