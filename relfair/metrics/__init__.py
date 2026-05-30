"""
relfair.metrics — Fairness metrics for regulatory audit.

Primary entry point:
    from relfair.metrics import compute_ll144_metrics, LL144Result
"""

from .ll144 import (
    EEOC_RACE_CATEGORIES,
    GroupStat,
    IntersectionalStat,
    LL144Result,
    compute_ll144_metrics,
)
from .stats import (
    FOUR_FIFTHS_THRESHOLD,
    SMALL_SAMPLE_THRESHOLD,
    bootstrap_ci,
    bootstrap_impact_ratio_cis,
    fisher_exact_pvalue,
    two_proportion_ztest_pvalue,
)

__all__ = [
    "compute_ll144_metrics",
    "LL144Result",
    "GroupStat",
    "IntersectionalStat",
    "EEOC_RACE_CATEGORIES",
    "FOUR_FIFTHS_THRESHOLD",
    "SMALL_SAMPLE_THRESHOLD",
    "bootstrap_ci",
    "bootstrap_impact_ratio_cis",
    "fisher_exact_pvalue",
    "two_proportion_ztest_pvalue",
]
