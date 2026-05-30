"""
Adult/Census Income — False-Positive & False-Negative Benchmark
================================================================
The canonical demonstration of BOTH failure modes of naive flip testing.

The setup
---------
`relationship=Husband` is sex-coded: 12,462 Male rows have it; exactly 1 Female
row does (data noise).  When naive testing flips sex Male->Female on a Husband
row, it produces (Female, Husband) — an impossible input the model has never
seen.

Two failure modes, both quantified here:

  1. FALSE POSITIVES  (theoretical / depends on model class)
     Model extrapolates on (Female, Husband) and may flip for the wrong reason.
     Detected via graph constraint-violation: naive CF violates the Husband->Wife
     hard rule -> definitionally off-manifold -> any flip is suspect.

  2. FALSE NEGATIVES  (empirically dominant with HistGBM)
     By leaving relationship=Husband, the naive test doesn't change the proxy
     feature the model actually uses.  Rel-aware swaps Husband->Wife, exposing
     the model's reliance on the marital-role proxy for income prediction.
     Result: rel-aware detects 3x more discrimination on Husband rows.

Both effects are real and important.  The empirically dominant one with modern
ML models is FALSE NEGATIVES.  The false-positive story is clearest with a
k-NN or a synthetic SCM (future experiment).

Usage
-----
    python run.py                 # standard (80/20 split)
    python run.py --seed 7        # different split
    python run.py --n-boot 2000   # tighter confidence intervals
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

from relfair.counterfactual import (
    batch_counterfactuals,
    detect_constraint_violations,
    flip_rate,
)
from relfair.graph import DependencyGraph
from relfair.manifold import ManifoldFilter
from relfair.mechanisms import LearnedMechanisms

from config import (
    CATEGORICAL_COLS, CONTINUOUS_COLS, EDGES,
    FEATURE_COLS, HARD_RULES, INTERVENTION, TARGET_COL,
)

RESULTS_DIR = Path(__file__).parent / "results"
DATA_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
TEST_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
DATA_PATH = Path(__file__).parent / "adult.parquet"


# ── helpers ──────────────────────────────────────────────────────────────────

def _tick(label: str) -> float:
    print(f"  {label}...", end=" ", flush=True)
    return time.time()

def _tock(t0: float) -> None:
    print(f"{time.time()-t0:.1f}s")


# ── data ─────────────────────────────────────────────────────────────────────

_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "_label",
]

def _read_url(url: str, skip_rows: int = 0) -> pd.DataFrame:
    raw = urllib.request.urlopen(url).read().decode()
    df = pd.read_csv(
        io.StringIO(raw), names=_COLS, skipinitialspace=True,
        na_values="?", skiprows=skip_rows,
    )
    df = df.dropna().copy()
    df["label"] = (
        df["_label"].str.strip().str.replace(".", "", regex=False) == ">50K"
    ).astype(int)
    return df.drop(columns="_label")

def load_data() -> pd.DataFrame:
    if DATA_PATH.exists():
        return pd.read_parquet(DATA_PATH)
    t0 = _tick("Downloading Adult dataset (train + test split)")
    train = _read_url(DATA_URL)
    test  = _read_url(TEST_URL, skip_rows=1)
    df = pd.concat([train, test], ignore_index=True)
    df.to_parquet(DATA_PATH)
    _tock(t0)
    return df

def split_data(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split
    return train_test_split(df, test_size=0.20, random_state=seed, stratify=df[TARGET_COL])


# ── model stack ───────────────────────────────────────────────────────────────

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

def fit_classifier(df_train: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder
    t0 = _tick("Fitting classifier (HistGBM)")
    pre = ColumnTransformer([
        ("num", "passthrough", CONTINUOUS_COLS),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_COLS),
    ], remainder="drop", sparse_threshold=0)
    clf = Pipeline([("pre", pre), ("clf", HistGradientBoostingClassifier(
        max_iter=200, random_state=0, early_stopping=False
    ))])
    clf.fit(df_train[FEATURE_COLS], df_train[TARGET_COL])
    _tock(t0)
    return clf

def fit_manifold(df_train: pd.DataFrame) -> ManifoldFilter:
    t0 = _tick("Fitting IsolationForest manifold filter")
    mf = ManifoldFilter(contamination=0.05, random_state=0)
    mf.fit(df_train[FEATURE_COLS])
    _tock(t0)
    return mf

def make_predict_fn(clf):
    def predict(df: pd.DataFrame) -> np.ndarray:
        return clf.predict(df[FEATURE_COLS])
    return predict


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(
    arr: np.ndarray, n_boot: int, ci: float, rng: np.random.Generator
) -> tuple[float, float]:
    if len(arr) == 0:
        return (0.0, 0.0)
    boot = [arr[rng.integers(0, len(arr), len(arr))].mean() for _ in range(n_boot)]
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))
    return lo, hi


# ── segment experiment ────────────────────────────────────────────────────────

def run_segment(
    subset: pd.DataFrame,
    G: DependencyGraph,
    mechs: LearnedMechanisms,
    mf: ManifoldFilter,
    predict_fn,
    *,
    attr: str,
    to_val: str,
    n_boot: int,
    seed: int,
    label: str,
) -> dict:
    rng = np.random.default_rng(seed + 77777)
    n = len(subset)

    naive    = batch_counterfactuals(subset, attr, to_val, G, mechs, naive=True,  seed=seed)
    relaware = batch_counterfactuals(subset, attr, to_val, G, mechs, naive=False, seed=seed)

    # --- Graph-based constraint violation (deterministic, primary metric) ---
    constraint_violated = detect_constraint_violations(naive, G)   # True = incoherent

    # --- IsolationForest (statistical, secondary) ---
    naive_cf_df = pd.DataFrame([r.counterfactual for r in naive])
    if_off      = ~mf.is_on_manifold(naive_cf_df[FEATURE_COLS])

    # Predictions
    orig_df  = pd.DataFrame([r.original      for r in naive])
    ra_df    = pd.DataFrame([r.counterfactual for r in relaware])
    pred_orig  = predict_fn(orig_df)
    pred_naive = predict_fn(naive_cf_df)
    pred_ra    = predict_fn(ra_df)

    flip_naive_all = pred_orig != pred_naive
    flip_ra_all    = pred_orig != pred_ra

    # Constrained subset: rows where naive CF is DEFINITELY off-manifold
    cv_mask = constraint_violated
    flip_naive_cv = flip_naive_all[cv_mask]
    flip_ra_cv    = flip_ra_all[cv_mask]

    def ci(arr):
        return bootstrap_ci(arr.astype(float), n_boot, 0.95, rng)

    return {
        "label": label,
        "n": n,
        # Constraint-violation stats
        "n_constraint_violated": int(cv_mask.sum()),
        "pct_constraint_violated": float(cv_mask.mean()),
        # IsolationForest stats (for comparison)
        "pct_if_off": float(if_off.mean()),
        # All-CFs rates
        "naive_flip_all":    float(flip_naive_all.mean()),
        "naive_flip_all_ci": ci(flip_naive_all),
        "ra_flip_all":       float(flip_ra_all.mean()),
        "ra_flip_all_ci":    ci(flip_ra_all),
        # Constraint-violated subset rates (false-positive analysis)
        "naive_flip_cv":    float(flip_naive_cv.mean()) if len(flip_naive_cv) else 0.0,
        "naive_flip_cv_ci": ci(flip_naive_cv),
        "ra_flip_cv":       float(flip_ra_cv.mean()) if len(flip_ra_cv) else 0.0,
        "ra_flip_cv_ci":    ci(flip_ra_cv),
        "fpr_reduction": (
            float(flip_naive_cv.mean() - flip_ra_cv.mean())
            if len(flip_naive_cv) else 0.0
        ),
        # Detection lift (false-negative story)
        "detection_lift": float(flip_ra_all.mean()) - float(flip_naive_all.mean()),
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: list[dict], out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    BG, PANEL = "#F5F1E8", "#FCF8F2"
    INK, INK2, HAIR = "#14130F", "#4A463E", "#D8D3C5"
    FLAG, PASS_ = "#B0411B", "#2D5F3F"

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=BG)
    fig.patch.set_facecolor(BG)

    titles = [
        "All male rows\n(combined)",
        "Husband rows\n(100% constraint-violated naive CFs)",
        "Non-Husband rows\n(control — no incoherence)",
    ]

    for ax, r, title in zip(axes, results, titles):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color=INK, fontsize=10.5, fontweight="bold", pad=8)
        ax.spines[:].set_color(HAIR)
        ax.yaxis.grid(True, color=HAIR, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2)

        w = 0.30
        x = np.array([0, 1])

        # Bars: All CFs (lighter) + Constraint-violated subset (solid)
        vals_all = [r["naive_flip_all"] * 100, r["ra_flip_all"] * 100]
        vals_cv  = [r["naive_flip_cv"]  * 100, r["ra_flip_cv"]  * 100]

        err_all = [
            [vals_all[j] - np.array(r[f"{'naive' if j==0 else 'ra'}_flip_all_ci"])[0]*100 for j in range(2)],
            [np.array(r[f"{'naive' if j==0 else 'ra'}_flip_all_ci"])[1]*100 - vals_all[j] for j in range(2)],
        ]
        err_cv = [
            [vals_cv[j]  - np.array(r[f"{'naive' if j==0 else 'ra'}_flip_cv_ci"])[0]*100  for j in range(2)],
            [np.array(r[f"{'naive' if j==0 else 'ra'}_flip_cv_ci"])[1]*100 - vals_cv[j]  for j in range(2)],
        ]

        ax.bar(x - w/2, vals_all, width=w, color=[FLAG, PASS_], alpha=0.40, zorder=3)
        ax.errorbar(x - w/2, vals_all, yerr=err_all, fmt="none", color=INK2,
                    capsize=3, linewidth=1.0, zorder=4)

        ax.bar(x + w/2, vals_cv, width=w, color=[FLAG, PASS_], alpha=1.0, zorder=3)
        ax.errorbar(x + w/2, vals_cv, yerr=err_cv, fmt="none", color=INK,
                    capsize=3.5, linewidth=1.5, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(["Naive", "Rel-aware"], color=INK2, fontsize=10)
        ax.set_ylabel("Flip rate (%)", color=INK2, fontsize=9)
        top = max(max(vals_all), max(vals_cv)) * 1.45 + 2
        ax.set_ylim(0, max(top, 6))

        lift = r["detection_lift"] * 100
        cv   = r["pct_constraint_violated"] * 100
        fpr  = r["fpr_reduction"] * 100

        ax.text(
            0.97, 0.97,
            f"n={r['n']:,}  |  constraint-violated: {cv:.0f}%\n"
            f"Detection lift: {lift:+.1f} pp\n"
            f"FPR reduction: {fpr:+.1f} pp",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            color=INK2, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=PANEL, ec=HAIR, lw=0.8),
        )

    legend_patches = [
        mpatches.Patch(color=FLAG, alpha=0.40, label="Naive  (all CFs)"),
        mpatches.Patch(color=FLAG, alpha=1.00, label="Naive  (constraint-violated)"),
        mpatches.Patch(color=PASS_, alpha=0.40, label="Rel-aware (all CFs)"),
        mpatches.Patch(color=PASS_, alpha=1.00, label="Rel-aware (same rows)"),
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.9, edgecolor=HAIR)

    fig.suptitle(
        "Adult/Census Income — Naive flip testing: two failure modes\n"
        "Husband rows: 100% constraint-violated naive CFs vs. coherent rel-aware (Female, Wife) CFs",
        color=INK, fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"\n  Chart saved -> {out_path}")


# ── print summary ─────────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    sep = "=" * 68
    print(f"\n{sep}")
    print("  Adult/Census Income -- Benchmark Results")
    print(sep)
    for r in results:
        cv_pct = r["pct_constraint_violated"] * 100
        print(f"\n  [{r['label']}]  n={r['n']:,}")
        print(f"    Constraint-violated naive CFs : {r['n_constraint_violated']:>5,} ({cv_pct:.0f}%)")
        print(f"    IsolationForest off-manifold  : {r['pct_if_off']:.1%}  (baseline comparison)")
        print()
        print(f"    All CFs:                naive={r['naive_flip_all']:6.1%}  rel-aware={r['ra_flip_all']:6.1%}")
        print(f"    Constraint-violated CFs:naive={r['naive_flip_cv']:6.1%}  rel-aware={r['ra_flip_cv']:6.1%}")
        lift = r["detection_lift"] * 100
        fpr  = r["fpr_reduction"] * 100
        print(f"    Detection lift (false-neg story): {lift:+.1f} pp  "
              f"{'<-- rel-aware detects more' if lift > 0 else ''}")
        print(f"    FPR reduction (false-pos story):  {fpr:+.1f} pp  "
              f"{'<-- POSITIVE: constraint fixing reduces flips' if fpr > 0 else '<-- rel-aware detects more (see ACS story)'}")
    print(f"\n{sep}")
    print()
    print("  KEY FINDING:")
    h = next(r for r in results if "Husband" in r["label"])
    print(f"  On the {h['n']:,} Husband rows, 100% of naive Male->Female CFs are")
    print(f"  constraint-violated: (Female, Husband) appears ~0 times in training.")
    print(f"  Naive testing detects {h['naive_flip_all']:.1%} discrimination.")
    print(f"  Relationship-aware testing detects {h['ra_flip_all']:.1%} -- {h['detection_lift']*100:+.1f} pp more.")
    print(f"  The naive test MISSES {h['detection_lift']*100:.1f} pp of discrimination")
    print(f"  because it leaves `relationship=Husband` unchanged, preventing the")
    print(f"  model from seeing the sex-driven marital-role income gap.")
    print(f"\n{sep}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    print("=" * 68)
    print("  relfair -- Adult/Census Income Benchmark")
    print("=" * 68)
    print(f"  Seed: {args.seed}  |  Bootstrap: {args.n_boot}")
    print()

    df = load_data()
    df_train, df_test = split_data(df, args.seed)
    print(f"  Train: {len(df_train):,}  |  Test: {len(df_test):,}")

    male_test    = df_test[df_test["sex"] == "Male"]
    husband_test = male_test[male_test["relationship"] == "Husband"]
    other_test   = male_test[male_test["relationship"] != "Husband"]
    print(f"  Male test rows:            {len(male_test):>5,}")
    print(f"    - relationship=Husband:  {len(husband_test):>5,}  ({len(husband_test)/len(male_test):.1%})")
    print(f"    - other relationships:   {len(other_test):>5,}")
    print()

    G     = build_graph()
    mechs = fit_mechanisms(G, df_train)
    mf    = fit_manifold(df_train)
    clf   = fit_classifier(df_train)
    pred  = make_predict_fn(clf)

    attr, to_val = INTERVENTION["attr"], INTERVENTION["to_val"]

    segments = [
        ("All male rows",    male_test.reset_index(drop=True)),
        ("Husband rows",     husband_test.reset_index(drop=True)),
        ("Non-Husband rows", other_test.reset_index(drop=True)),
    ]

    all_results = []
    for seg_label, seg_df in segments:
        print(f"  Running segment: {seg_label} (n={len(seg_df):,})...")
        t0 = time.time()
        r = run_segment(
            seg_df, G, mechs, mf, pred,
            attr=attr, to_val=to_val,
            n_boot=args.n_boot, seed=args.seed,
            label=seg_label,
        )
        cv = r["pct_constraint_violated"] * 100
        lift = r["detection_lift"] * 100
        print(f"    constraint-violated={cv:.0f}%  detection-lift={lift:+.1f}pp  [{time.time()-t0:.1f}s]")
        all_results.append(r)

    print_summary(all_results)
    plot_results(all_results, RESULTS_DIR / "fpr_reduction_adult.png")


if __name__ == "__main__":
    main()
