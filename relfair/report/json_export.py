"""
Machine-readable JSON export of LL144Result.

The output is content-addressed (SHA-256 of the input data hash is embedded)
so each report is uniquely identified and reproducible.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from relfair.metrics.ll144 import GroupStat, IntersectionalStat, LL144Result


def _group_stat_to_dict(s: GroupStat) -> dict[str, Any]:
    return {
        "group":            s.group,
        "n":                s.n,
        "selected":         s.selected,
        "rate":             round(s.rate, 6),
        "impact_ratio":     round(s.ratio, 6),
        "impact_ratio_ci":  [round(s.ratio_ci_lo, 6), round(s.ratio_ci_hi, 6)],
        "four_fifths_flag": s.four_fifths_flag,
        "p_value":          round(s.p_value, 6) if s.p_value is not None else None,
        "small_sample":     s.small_sample,
        "is_reference":     s.is_reference,
    }


def _cell_to_dict(c: IntersectionalStat) -> dict[str, Any]:
    return {
        "sex":              c.sex,
        "race":             c.race,
        "n":                c.n,
        "selected":         c.selected,
        "rate":             round(c.rate, 6),
        "impact_ratio":     round(c.ratio, 6),
        "four_fifths_flag": c.four_fifths_flag,
        "small_sample":     c.small_sample,
    }


def to_dict(result: LL144Result) -> dict[str, Any]:
    """Convert an LL144Result to a JSON-serialisable dict."""
    return {
        "schema_version": "1.0",
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "meta": result.meta,
        "summary": {
            "n_total":      result.n_total,
            "n_selected":   result.n_selected,
            "overall_rate": round(result.overall_rate, 6),
            "outcome_type": result.outcome_type,
            "threshold":    result.threshold,
        },
        "by_sex":  [_group_stat_to_dict(s) for s in result.by_sex],
        "by_race": [_group_stat_to_dict(s) for s in result.by_race],
        "intersectional": [_cell_to_dict(c) for c in result.intersectional],
        "flags": {
            "by_sex":         [s.group for s in result.by_sex  if s.four_fifths_flag],
            "by_race":        [s.group for s in result.by_race if s.four_fifths_flag],
            "intersectional": [
                f"{c.sex} × {c.race}"
                for c in result.intersectional
                if c.four_fifths_flag
            ],
        },
    }


def to_json(result: LL144Result, indent: int = 2) -> str:
    """Serialise an LL144Result to a pretty-printed JSON string."""
    return json.dumps(to_dict(result), indent=indent, ensure_ascii=False)


def write_json(result: LL144Result, path: str | Path, indent: int = 2) -> None:
    """Write the JSON export to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(result, indent=indent), encoding="utf-8")
