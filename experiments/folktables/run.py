"""
Folktables ACS Income — FPR Reduction Benchmark
================================================
Measures how relationship-aware counterfactual generation reduces the false-
positive rate of sex-fairness testing vs. naive attribute-flip testing.

Headline metric
---------------
    FPR_reduction = naive_flip_rate − relaware_flip_rate
                    (restricted to naive CFs the manifold filter rejects)

Usage
-----
    python run.py                           # full run (50k train, 10k test)
    python run.py --n-train 5000 --n-test 2000   # quick debug run
    python run.py --states CA TX NY         # multi-state
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

# Ensure Unicode output works on Windows consoles (cp1252 / cp850)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

# ── resolve package root ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from relfair.counterfactual import batch_counterfactuals, flip_rate
from relfair.graph import DependencyGraph
from relfair.manifold import ManifoldFilter
from relfair.mechanisms import LearnedMechanisms

from config import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    EDGES,
    FEATURE_COLS,
    HARD_RULES,
    INTERVENTION,
    TARGET_COL,
)

RESULTS_DIR = Path(__file__).parent / "results"


# ── helpers ─────────────────────────────────────────────────────────────────

def _tick(label: str) -> float:
    print(f"  {label}...", end=" ", flush=True)
    return time.time()


def _tock(t0: float) -> None:
    print(f"{time.time() - t0:.1f}s")


# ── data ────────────────────────────────────────────────────────────────────

def load_acs(states: list[str]) -> pd.DataFrame:
    try:
        from folktables import ACSDataSource, ACSIncome
    except ImportError:
        raise ImportError("pip install folktables")

    t0 = _tick(f"Loading ACS data ({', '.join(states)})")
    ds = ACSDataSource(survey_year="2018", horizon="1-Year", survey="person")
    raw = ds.get_data(states=states, download=True)
    X, y, _ = ACSIncome.df_to_numpy(raw)
    df = pd.DataFrame(X, columns=FEATURE_COLS)
    df[TARGET_COL] = y.astype(int)
    _tock(t0)
    return df


def split_data(
    df: pd.DataFrame,
    n_train: int,
    n_test: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split keeping only rows where SEX==from_val in test."""
    from sklearn.model_selection import train_test_split

    df = df.sample(n=min(n_train + n_test, len(df)), random_state=seed).reset_index(drop=True)
    df_train, df_test = train_test_split(
        df, train_size=n_train, test_size=n_test, random_state=seed
    )
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)


# ── graph + mechanisms ───────────────────────────────────────────────────────

def build_graph() -> DependencyGraph:
    G = DependencyGraph.from_edges(EDGES)
    for rule in HARD_RULES:
        G.add_hard_rule(**rule)
    return G


def fit_mechanisms(G: DependencyGraph, df_train: pd.DataFrame) -> LearnedMechanisms:
    t0 = _tick("Fitting mechanisms")
    mechs = LearnedMechanisms(G)
    mechs.fit(df_train[FEATURE_COLS], cat_cols=CATEGORICAL_COLS)
    _tock(t0)
    return mechs


# ── classifier ───────────────────────────────────────────────────────────────

def fit_classifier(df_train: pd.DataFrame):
    """HistGBM classifier with OrdinalEncoder for categorical features."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    t0 = _tick("Fitting classifier")
    pre = ColumnTransformer([
        ("num", "passthrough", CONTINUOUS_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
    ], remainder="drop", sparse_threshold=0)

    clf = Pipeline([
        ("pre", pre),
        ("clf", HistGradientBoostingClassifier(max_iter=100, random_state=0)),
    ])
    clf.fit(df_train[FEATURE_COLS], df_train[TARGET_COL])
    _tock(t0)
    return clf


def make_predict_fn(clf, feature_cols: list[str], cat_cols: list[str]):
    """Wrap classifier.predict so it handles mixed-type CF DataFrames."""
    def predict(df: pd.DataFrame) -> np.ndarray:
        X = df[feature_cols].copy()
        for c in cat_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce")
        return clf.predict(X)
    return predict


# ── manifold filter ──────────────────────────────────────────────────────────

def fit_manifold(df_train: pd.DataFrame) -> ManifoldFilter:
    t0 = _tick("Fitting manifold filter")
    # Use numeric representation of all features for IsolationForest
    X = df_train[FEATURE_COLS].copy()
    for c in CATEGORICAL_COLS:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    mf = ManifoldFilter(contamination=0.05, random_state=0)
    mf.fit(X)
    _tock(t0)
    return mf


def score_manifold(mf: ManifoldFilter, cf_records, feature_cols: list[str], cat_cols: list[str]) -> np.ndarray:
    cf_df = pd.DataFrame([r.counterfactual for r in cf_records])
    X = cf_df[feature_cols].copy()
    for c in cat_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return mf.is_on_manifold(X)


# ── experiment ───────────────────────────────────────────────────────────────

def run_intervention(
    df_test: pd.DataFrame,
    G: DependencyGraph,
    mechs: LearnedMechanisms,
    mf: ManifoldFilter,
    predict_fn,
    *,
    attr: str,
    from_val: float,
    to_val: float,
    label: str,
    seed: int,
) -> dict:
    """Run one protected-attribute intervention on the test rows where attr==from_val."""

    # Only flip rows where attr == from_val (e.g. Male rows for Male→Female)
    subset = df_test[df_test[attr] == from_val].reset_index(drop=True)
    n = len(subset)
    print(f"\n  Intervention: {label}  |  n={n:,} rows")

    t0 = _tick("  Naive CFs")
    naive = batch_counterfactuals(subset, attr, to_val, G, mechs, naive=True, seed=seed)
    _tock(t0)

    t0 = _tick("  Rel-aware CFs")
    relaware = batch_counterfactuals(subset, attr, to_val, G, mechs, naive=False, seed=seed)
    _tock(t0)

    t0 = _tick("  Manifold scoring")
    on_manifold_naive = score_manifold(mf, naive, FEATURE_COLS, CATEGORICAL_COLS)
    on_manifold_relaware = score_manifold(mf, relaware, FEATURE_COLS, CATEGORICAL_COLS)
    off_mask = ~on_manifold_naive     # rows where naive CF is off-manifold
    _tock(t0)

    t0 = _tick("  Computing flip rates")
    r_naive_all      = flip_rate(naive,    predict_fn)
    r_relaware_all   = flip_rate(relaware, predict_fn)
    r_naive_off      = flip_rate(naive,    predict_fn, on_manifold_mask=off_mask.tolist())
    r_relaware_off   = flip_rate(relaware, predict_fn, on_manifold_mask=off_mask.tolist())
    _tock(t0)

    fpr_reduction = r_naive_off["flip_rate"] - r_relaware_off["flip_rate"]

    return {
        "label": label,
        "n_rows": n,
        "n_offmanifold_naive": int(off_mask.sum()),
        "pct_offmanifold_naive": float(off_mask.mean()),
        "pct_offmanifold_relaware": float((~on_manifold_relaware).mean()),
        "naive_flip_rate_all":      r_naive_all["flip_rate"],
        "relaware_flip_rate_all":   r_relaware_all["flip_rate"],
        "naive_flip_rate_offmanifold":    r_naive_off["flip_rate"],
        "relaware_flip_rate_offmanifold": r_relaware_off["flip_rate"],
        "fpr_reduction": fpr_reduction,
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Design tokens from FairLens brand
    BG      = "#F5F1E8"
    PANEL   = "#FCF8F2"
    INK     = "#14130F"
    INK2    = "#4A463E"
    FLAG    = "#B0411B"   # naive (bad)
    PASS    = "#2D5F3F"   # rel-aware (good)
    HAIR    = "#D8D3C5"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    # ── left: all-CFs flip rates ──────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(PANEL)
    bars = ax.bar(
        ["Naive", "Rel-aware"],
        [results["naive_flip_rate_all"] * 100, results["relaware_flip_rate_all"] * 100],
        color=[FLAG, PASS], width=0.5, zorder=3,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=4, color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel("Flip rate (%)", color=INK2, fontsize=10)
    ax.set_title("Flip rate — all counterfactuals", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(results["naive_flip_rate_all"] * 120, 5))
    ax.tick_params(colors=INK2)
    ax.spines[:].set_color(HAIR)
    ax.yaxis.grid(True, color=HAIR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # ── right: off-manifold CFs only (the FPR story) ──────────────────────
    ax = axes[1]
    ax.set_facecolor(PANEL)
    bars2 = ax.bar(
        ["Naive\n(off-manifold)", "Rel-aware\n(same rows)"],
        [
            results["naive_flip_rate_offmanifold"] * 100,
            results["relaware_flip_rate_offmanifold"] * 100,
        ],
        color=[FLAG, PASS], width=0.5, zorder=3,
    )
    ax.bar_label(bars2, fmt="%.1f%%", padding=4, color=INK, fontsize=11, fontweight="bold")
    fpr_pct = results["fpr_reduction"] * 100
    ax.set_title(
        f"Flip rate — off-manifold CFs only\n"
        f"FPR reduction: {fpr_pct:+.1f} pp  "
        f"({results['n_offmanifold_naive']:,} / {results['n_rows']:,} rows = "
        f"{results['pct_offmanifold_naive']:.1%} off-manifold)",
        color=INK, fontsize=10, fontweight="bold",
    )
    ax.set_ylim(0, max(results["naive_flip_rate_offmanifold"] * 130, 5))
    ax.tick_params(colors=INK2)
    ax.spines[:].set_color(HAIR)
    ax.yaxis.grid(True, color=HAIR, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)

    # annotation arrow
    lo = results["relaware_flip_rate_offmanifold"] * 100
    hi = results["naive_flip_rate_offmanifold"] * 100
    if hi - lo > 1:
        ax.annotate(
            "",
            xy=(1, lo + 1), xytext=(1, hi - 1),
            arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.5),
        )
        ax.text(1.28, (lo + hi) / 2, f"−{hi - lo:.1f} pp",
                va="center", ha="left", color=INK2, fontsize=9)

    fig.suptitle(
        "Relationship-aware counterfactual generation reduces fairness-testing false positives\n"
        "ACS Income (California 2018) — SEX: Male → Female",
        color=INK, fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Chart saved → {out_path}")


# ── print summary ─────────────────────────────────────────────────────────────

def print_summary(results: dict) -> None:
    r = results
    sep = "─" * 56
    print(f"\n{sep}")
    print("  ACS Income — FPR Reduction Benchmark")
    print(sep)
    print(f"  Intervention          : {r['label']}")
    print(f"  Test rows             : {r['n_rows']:>8,}")
    print(f"  Off-manifold (naive)  : {r['n_offmanifold_naive']:>8,}  ({r['pct_offmanifold_naive']:.1%})")
    print(f"  Off-manifold (rel-aw) : {r['pct_offmanifold_relaware']:.1%}")
    print()
    print("  ── All counterfactuals ─────────────────────────────")
    print(f"  Naive flip rate       : {r['naive_flip_rate_all']:>7.1%}")
    print(f"  Rel-aware flip rate   : {r['relaware_flip_rate_all']:>7.1%}")
    print()
    print("  ── Off-manifold CFs only (the FPR story) ───────────")
    print(f"  Naive flip rate       : {r['naive_flip_rate_offmanifold']:>7.1%}")
    print(f"  Rel-aware flip rate   : {r['relaware_flip_rate_offmanifold']:>7.1%}")
    print(f"  FPR reduction         : {r['fpr_reduction']:>+7.1%}")
    print(sep)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Folktables FPR-reduction benchmark")
    parser.add_argument("--n-train", type=int, default=50_000, metavar="N",
                        help="Training rows for mechanisms + classifier (default: 50000)")
    parser.add_argument("--n-test",  type=int, default=10_000, metavar="N",
                        help="Test rows for evaluation (default: 10000)")
    parser.add_argument("--states", nargs="+", default=["CA"],
                        help="State codes (default: CA)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 56)
    print("  relfair — FPR Reduction Benchmark (Folktables ACS)")
    print("=" * 56)
    print(f"  Train: {args.n_train:,}  |  Test: {args.n_test:,}  |  Seed: {args.seed}")
    print(f"  States: {args.states}")
    print()

    # 1. Data
    df = load_acs(args.states)
    df_train, df_test = split_data(df, args.n_train, args.n_test, args.seed)
    print(f"  Train: {len(df_train):,} rows  |  Test: {len(df_test):,} rows")

    # 2. Graph + mechanisms (fit on train only)
    G = build_graph()
    mechs = fit_mechanisms(G, df_train)

    # 3. Manifold filter (fit on train only)
    mf = fit_manifold(df_train)

    # 4. Classifier (fit on train only)
    clf = fit_classifier(df_train)
    predict_fn = make_predict_fn(clf, FEATURE_COLS, CATEGORICAL_COLS)

    # 5. Run the Sex → Female intervention
    results = run_intervention(
        df_test, G, mechs, mf, predict_fn,
        **INTERVENTION,
        seed=args.seed,
    )

    # 6. Report + chart
    print_summary(results)
    plot_results(results, RESULTS_DIR / "fpr_reduction_acs_income.png")


if __name__ == "__main__":
    main()
