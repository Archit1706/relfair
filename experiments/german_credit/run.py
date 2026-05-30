"""
German Credit — False-Positive Reduction Benchmark
===================================================
Demonstrates that naive attribute-flip testing generates FALSE POSITIVES
on the German Credit dataset because the `personal_status` column encodes
BOTH sex AND marital status in a single categorical field.

  Values: "male div/sep" | "male single" | "male mar/wid" | "female div/dep/mar"

Naive intervention (Male -> Female):
  Flips only personal_status to "female div/dep/mar" but leaves co-occurring
  male-coded credit attributes (credit_amount, duration) unchanged.
  The result is an off-manifold input: a "female" profile with male-typical
  borrowing patterns. The model may flip its prediction on this incoherent
  input — a FALSE POSITIVE (discrimination signal from junk input, not real bias).

Relationship-aware intervention:
  Applies the hard rule (personal_status remapping) then re-derives
  credit_amount and duration from their conditional distributions given
  the female personal_status. The result is a coherent, on-manifold female
  profile. Fewer spurious flips -> lower false-positive rate.

Usage
-----
    python run.py              # standard run (800 train / 200 test)
    python run.py --seed 7     # different random split
"""

from __future__ import annotations

import argparse
import io
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from relfair.counterfactual import batch_counterfactuals, flip_rate
from relfair.graph import DependencyGraph
from relfair.manifold import ManifoldFilter
from relfair.mechanisms import LearnedMechanisms

from config import (
    CATEGORICAL_COLS,
    EDGES,
    FEATURE_COLS,
    HARD_RULES,
    INTERVENTION,
    NUMERIC_COLS,
    TARGET_COL,
)

RESULTS_DIR = Path(__file__).parent / "results"
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_PATH = Path(__file__).parent / "german.data"


# ── helpers ─────────────────────────────────────────────────────────────────

def _tick(label: str) -> float:
    print(f"  {label}...", end=" ", flush=True)
    return time.time()


def _tock(t0: float) -> None:
    print(f"{time.time() - t0:.1f}s")


# ── data ────────────────────────────────────────────────────────────────────

# Column names in order for german.data (space-separated, no header)
_RAW_COLS = [
    "status", "duration", "credit_history", "purpose", "credit_amount",
    "savings", "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property", "age",
    "other_plans", "housing", "existing_credits", "job", "num_dependents",
    "telephone", "foreign_worker", "target_raw",
]

# UCI encoding -> readable strings for personal_status (attribute A13 = col 9)
_PS_MAP = {
    "A91": "male div/sep",
    "A92": "female div/dep/mar",
    "A93": "male single",
    "A94": "male mar/wid",
    "A95": "female single",   # very rare
}


def download_data() -> None:
    if DATA_PATH.exists():
        return
    print(f"  Downloading German Credit data from UCI...", end=" ", flush=True)
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print("done")


def load_data() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(DATA_PATH, sep=" ", header=None, names=_RAW_COLS)

    # Decode personal_status
    df["personal_status"] = df["personal_status"].map(_PS_MAP).fillna("unknown")

    # Target: UCI uses 1=good, 2=bad. Recode to 1=good, 0=bad.
    df[TARGET_COL] = (df["target_raw"] == 1).astype(int)
    df = df.drop(columns=["target_raw"])

    return df


def split_data(df: pd.DataFrame, n_train: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    return train_test_split(df, train_size=n_train, random_state=seed)


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
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    t0 = _tick("Fitting classifier")
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
    ], remainder="drop", sparse_threshold=0)

    clf = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=200, random_state=0, early_stopping=False
    ))])
    clf.fit(df_train[FEATURE_COLS], df_train[TARGET_COL])
    _tock(t0)
    return clf


def make_predict_fn(clf):
    def predict(df: pd.DataFrame) -> np.ndarray:
        return clf.predict(df[FEATURE_COLS])
    return predict


# ── manifold filter ──────────────────────────────────────────────────────────

def fit_manifold(df_train: pd.DataFrame) -> ManifoldFilter:
    t0 = _tick("Fitting manifold filter")
    mf = ManifoldFilter(contamination=0.05, random_state=0)
    mf.fit(df_train[FEATURE_COLS])
    _tock(t0)
    return mf


# ── experiment ───────────────────────────────────────────────────────────────

def run_intervention(
    df_test: pd.DataFrame,
    G: DependencyGraph,
    mechs: LearnedMechanisms,
    mf: ManifoldFilter,
    predict_fn,
    *,
    attr: str,
    from_vals: tuple,
    to_val: str,
    label: str,
    seed: int,
) -> dict:
    subset = df_test[df_test[attr].isin(from_vals)].reset_index(drop=True)
    n = len(subset)
    print(f"\n  Intervention: {label}  |  n={n} rows")

    t0 = _tick("  Naive CFs")
    naive = batch_counterfactuals(subset, attr, to_val, G, mechs, naive=True, seed=seed)
    _tock(t0)

    t0 = _tick("  Rel-aware CFs")
    relaware = batch_counterfactuals(subset, attr, to_val, G, mechs, naive=False, seed=seed)
    _tock(t0)

    # Manifold scoring — German Credit features are all string/numeric already
    naive_cf_df    = pd.DataFrame([r.counterfactual for r in naive])
    relaware_cf_df = pd.DataFrame([r.counterfactual for r in relaware])

    on_manifold_naive    = mf.is_on_manifold(naive_cf_df[FEATURE_COLS])
    on_manifold_relaware = mf.is_on_manifold(relaware_cf_df[FEATURE_COLS])
    off_mask = ~on_manifold_naive

    r_naive_all    = flip_rate(naive,    predict_fn)
    r_relaware_all = flip_rate(relaware, predict_fn)
    r_naive_off    = flip_rate(naive,    predict_fn, on_manifold_mask=off_mask.tolist())
    r_relaware_off = flip_rate(relaware, predict_fn, on_manifold_mask=off_mask.tolist())

    fpr_reduction = r_naive_off["flip_rate"] - r_relaware_off["flip_rate"]

    return {
        "label": label,
        "n_rows": n,
        "n_offmanifold_naive":    int(off_mask.sum()),
        "pct_offmanifold_naive":  float(off_mask.mean()),
        "pct_offmanifold_relaware": float((~on_manifold_relaware).mean()),
        "naive_flip_rate_all":       r_naive_all["flip_rate"],
        "relaware_flip_rate_all":    r_relaware_all["flip_rate"],
        "naive_flip_rate_offmanifold":    r_naive_off["flip_rate"],
        "relaware_flip_rate_offmanifold": r_relaware_off["flip_rate"],
        "fpr_reduction": fpr_reduction,
    }


# ── plotting ─────────────────────────────────────────────────────────────────

def plot_results(results: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BG, PANEL, INK, INK2 = "#F5F1E8", "#FCF8F2", "#14130F", "#4A463E"
    FLAG, PASS_, HAIR = "#B0411B", "#2D5F3F", "#D8D3C5"

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, (title, naive_val, relaware_val, note) in zip(axes, [
        (
            "Flip rate — all counterfactuals",
            results["naive_flip_rate_all"] * 100,
            results["relaware_flip_rate_all"] * 100,
            "",
        ),
        (
            "Flip rate — off-manifold CFs only",
            results["naive_flip_rate_offmanifold"] * 100,
            results["relaware_flip_rate_offmanifold"] * 100,
            f"FPR reduction: {results['fpr_reduction']*100:+.1f} pp\n"
            f"({results['n_offmanifold_naive']} / {results['n_rows']} rows off-manifold)",
        ),
    ]):
        ax.set_facecolor(PANEL)
        bars = ax.bar(
            ["Naive", "Rel-aware"], [naive_val, relaware_val],
            color=[FLAG, PASS_], width=0.5, zorder=3,
        )
        ax.bar_label(bars, fmt="%.1f%%", padding=4, color=INK, fontsize=11, fontweight="bold")
        ax.set_ylabel("Flip rate (%)", color=INK2, fontsize=10)
        ax.set_title(f"{title}\n{note}", color=INK, fontsize=10, fontweight="bold")
        ax.set_ylim(0, max(naive_val * 1.4, 10))
        ax.tick_params(colors=INK2)
        ax.spines[:].set_color(HAIR)
        ax.yaxis.grid(True, color=HAIR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)

    fig.suptitle(
        "German Credit — Naive testing creates false positives via incoherent personal_status inputs\n"
        "Relationship-aware generation produces coherent female profiles, reducing the false-positive rate",
        color=INK, fontsize=11, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Chart saved -> {out_path}")


# ── print summary ─────────────────────────────────────────────────────────────

def print_summary(r: dict) -> None:
    sep = "-" * 56
    print(f"\n{sep}")
    print("  German Credit -- FPR Reduction Benchmark")
    print(sep)
    print(f"  Intervention          : {r['label']}")
    print(f"  Test rows             : {r['n_rows']:>6}")
    print(f"  Off-manifold (naive)  : {r['n_offmanifold_naive']:>6}  ({r['pct_offmanifold_naive']:.1%})")
    print(f"  Off-manifold (rel-aw) : {r['pct_offmanifold_relaware']:.1%}")
    print()
    print("  -- All counterfactuals --------------------------------")
    print(f"  Naive flip rate       : {r['naive_flip_rate_all']:>7.1%}")
    print(f"  Rel-aware flip rate   : {r['relaware_flip_rate_all']:>7.1%}")
    print()
    print("  -- Off-manifold CFs only (false-positive story) ------")
    print(f"  Naive flip rate       : {r['naive_flip_rate_offmanifold']:>7.1%}")
    print(f"  Rel-aware flip rate   : {r['relaware_flip_rate_offmanifold']:>7.1%}")
    print(f"  FPR reduction         : {r['fpr_reduction']:>+7.1%}")
    print(sep)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="German Credit FPR-reduction benchmark")
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--seed",    type=int, default=42)
    args = parser.parse_args()

    print("=" * 56)
    print("  relfair -- FPR Reduction Benchmark (German Credit)")
    print("=" * 56)

    df = load_data()
    print(f"  Loaded {len(df)} rows, {df['personal_status'].value_counts().to_dict()}")

    df_train, df_test = split_data(df, args.n_train, args.seed)
    print(f"  Train: {len(df_train)}  |  Test: {len(df_test)}")

    G    = build_graph()
    mechs = fit_mechanisms(G, df_train)
    mf   = fit_manifold(df_train)
    clf  = fit_classifier(df_train)
    pred = make_predict_fn(clf)

    results = run_intervention(
        df_test, G, mechs, mf, pred,
        **INTERVENTION, seed=args.seed,
    )

    print_summary(results)
    plot_results(results, RESULTS_DIR / "fpr_reduction_german_credit.png")


if __name__ == "__main__":
    main()
