"""
Statistical utilities for fairness metrics.

All functions are pure — no I/O, no global state, seed-explicit.
These back the bootstrap CIs and significance tests in ll144.py.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ── constants ────────────────────────────────────────────────────────────────

SMALL_SAMPLE_THRESHOLD = 30   # per EEOC guidance; flag when n < this
FOUR_FIFTHS_THRESHOLD  = 0.80  # NYC LL 144 §5-303 four-fifths rule


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(
    data: pd.DataFrame,
    stat_fn: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap confidence interval for any scalar statistic.

    Resamples *data* with replacement, applies *stat_fn* to each resample,
    and returns the (lo, hi) percentile CI bounds.

    Args:
        data:    DataFrame to resample (rows are observations).
        stat_fn: Function mapping a DataFrame to a scalar.
        n_boot:  Number of bootstrap resamples (2 000 is standard).
        ci:      Coverage level, e.g. 0.95 for 95% CI.
        seed:    RNG seed for reproducibility.

    Returns:
        (lo, hi) — lower and upper bounds of the CI.
    """
    rng = np.random.default_rng(seed)
    n   = len(data)
    boot: list[float] = []
    for _ in range(n_boot):
        idx     = rng.integers(0, n, n)
        resample = data.iloc[idx]
        try:
            boot.append(float(stat_fn(resample)))
        except (ZeroDivisionError, ValueError):
            # Empty group in resample — skip this draw
            pass
    if not boot:
        return (0.0, 0.0)
    alpha = (1.0 - ci) / 2.0
    return (
        float(np.percentile(boot, alpha * 100)),
        float(np.percentile(boot, (1 - alpha) * 100)),
    )


def bootstrap_impact_ratio_cis(
    df: pd.DataFrame,
    group_col: str,
    selected_col: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """
    Bootstrap CIs for impact ratios across all groups in *group_col*.

    Propagates uncertainty in the reference group's rate into each ratio.
    Returns {group_name: (ci_lo, ci_hi)}.
    """
    groups = df[group_col].unique()
    rng    = np.random.default_rng(seed)
    n      = len(df)

    # n_boot × n_groups matrix of selection rates
    rate_matrix: dict[str, list[float]] = {g: [] for g in groups}

    for _ in range(n_boot):
        idx      = rng.integers(0, n, n)
        resample = df.iloc[idx]
        rates: dict[str, float] = {}
        for g in groups:
            sub = resample[resample[group_col] == g]
            if len(sub) == 0:
                rates[g] = 0.0
            else:
                rates[g] = float(sub[selected_col].mean())
        max_rate = max(rates.values()) if rates else 1.0
        if max_rate == 0.0:
            max_rate = 1.0
        for g in groups:
            rate_matrix[g].append(rates[g] / max_rate)

    ci_out: dict[str, tuple[float, float]] = {}
    alpha = 0.025
    for g in groups:
        arr = np.array(rate_matrix[g])
        ci_out[g] = (
            float(np.percentile(arr, alpha * 100)),
            float(np.percentile(arr, (1 - alpha) * 100)),
        )
    return ci_out


# ── significance tests ────────────────────────────────────────────────────────

def fisher_exact_pvalue(
    df: pd.DataFrame,
    group_col: str,
    selected_col: str,
    group_a: str,
    group_b: str,
) -> float | None:
    """
    Two-sided Fisher's exact test: group_a vs group_b.

    Contingency table:
        [ selected_a    not_selected_a ]
        [ selected_b    not_selected_b ]

    Returns p-value, or None if either group is empty.
    """
    a = df[df[group_col] == group_a]
    b = df[df[group_col] == group_b]
    if len(a) == 0 or len(b) == 0:
        return None

    sel_a   = int(a[selected_col].sum())
    nsel_a  = len(a) - sel_a
    sel_b   = int(b[selected_col].sum())
    nsel_b  = len(b) - sel_b

    table = [[sel_a, nsel_a], [sel_b, nsel_b]]
    _, p  = scipy_stats.fisher_exact(table, alternative="two-sided")
    return float(p)


def two_proportion_ztest_pvalue(
    n_a: int, k_a: int,
    n_b: int, k_b: int,
) -> float | None:
    """
    Two-proportion z-test: proportion k_a/n_a vs k_b/n_b.
    Returns two-sided p-value, or None on degenerate input.
    """
    if n_a == 0 or n_b == 0:
        return None
    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)
    if p_pool in (0.0, 1.0):
        return None
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        return None
    z = (p_a - p_b) / se
    p = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    return p
