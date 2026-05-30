# Experiments — Benchmark Study

This directory contains the empirical study that is simultaneously the paper, the demo, and the product's headline metric.

---

## The unified claim

> Naive attribute-flip testing is unreliable in both directions:
> — it **understates discrimination** by not co-flipping proxy features
>   (false negatives)
> — it produces **incoherent inputs** that violate causal hard rules,
>   making any flip on those inputs methodologically suspect
>   (potential false positives, model-class dependent)
>
> Relationship-aware counterfactual generation addresses both by propagating
> interventions through the causal graph, producing internally consistent,
> on-manifold counterfactuals that accurately measure total discriminatory impact.

---

## Results summary

### Dataset 1 — Adult/Census Income (UCI, 45K rows)

**The canonical false-positive case: `relationship: Husband/Wife`**

`Husband=Female` never occurs in the training data (1 noisy row out of 30K).
Every naive Male→Female CF on a Husband row is definitionally constraint-violated.

| Segment | n | Constraint-violated | Naive flip | Rel-aware flip | Detection lift |
|---------|---|---------------------|------------|---------------|----------------|
| All male rows | 6,061 | 61% | 4.6% | 14.8% | **+10.2 pp** |
| Husband rows | 3,682 | **100%** | 7.0% | 24.3% | **+17.2 pp** |
| Non-Husband (control) | 2,379 | **0%** | 0.9% | 1.5% | **+0.6 pp** |

**Key findings:**
- Graph-based constraint detection: **100% recall** for Husband rows, **0% false positives** on non-Husband rows. IsolationForest flags 7.5% uniformly — it cannot see the specific incoherence.
- The +17.2 pp detection lift on Husband rows vs. +0.6 pp on non-Husband rows proves the effect is caused specifically by the Husband/Wife incoherence, not other features.
- Naive testing misses 17 pp of discrimination because `relationship=Husband` suppresses the model's response. Rel-aware testing exposes the sex-driven marital-role income gap.

---

### Dataset 2 — Folktables ACS Income (CA 2018, 195K rows)

**Soft dependencies: occupation, hours/week, marital status**

No hard rules. All sex dependencies are soft (learned mechanisms):
`SEX → OCCP, WKHP, MAR, COW, RELP`

| | Naive | Rel-aware | Lift |
|--|-------|-----------|------|
| All male→female CFs | 7.5% | 34.5% | **+27.0 pp** |
| Off-manifold naive CFs | 3.6% | 16.9% | +13.3 pp |

**Key finding:** Rel-aware testing detects **4.6× more discrimination** than naive testing. The model uses occupation (OCCP) as a sex proxy — male-dominated occupations predict higher income. Naive testing, by leaving OCCP unchanged, misses this entirely. Relationship-aware testing re-derives OCCP from female conditional distributions and exposes the proxy discrimination.

---

### Dataset 3 — German Credit (UCI, 1K rows)

**Hard rules on `personal_status` (sex + marital status encoded together)**

`personal_status` encodes both sex and marital status: `male single`, `male div/sep`, `male mar/wid`, `female div/dep/mar`.

| | Naive | Rel-aware | Lift |
|--|-------|-----------|------|
| All CFs (male variants → female) | 4.9% | 13.2% | **+8.3 pp** |

**Key finding:** Same false-negative pattern as ACS. Naive testing detects only 4.9% discrimination; rel-aware (re-deriving `credit_amount` and `duration` from female personal_status distributions) detects 13.2%. Small sample (144 test rows) limits statistical power.

---

## The key methodological contribution: graph-based vs. statistical off-manifold detection

| Method | Husband rows (should be 100%) | Non-Husband rows (should be 0%) |
|--------|-------------------------------|----------------------------------|
| IsolationForest | 7.5% | 7.4% |
| Graph constraint violation | **100%** | **0%** |

IsolationForest cannot identify specific attribute-value incoherences caused by hard-rule violations. It flags 7.5% of rows regardless of whether they involve Husband/Wife incoherence.

Graph-based constraint checking (`detect_constraint_violations`) provides **deterministic, 100% recall** for hard-rule violations. This is a clean methodological contribution: use graph-based checking for hard rules; use statistical filtering (IsolationForest, KDE) for soft distribution violations.

---

## Running the experiments

```bash
pip install -e "../../[experiments]"   # from relfair/ root

# Adult/Census (recommended: cleanest results)
cd adult && python run.py

# ACS Income (large scale, soft dependencies)
cd folktables && python run.py --n-train 50000 --n-test 10000

# German Credit (small dataset, hard rule on personal_status)
cd german_credit && python run.py
```

---

## Design choices

### Interventional vs. individual-level counterfactuals

Mechanisms are fitted on conditional distributions P(descendant | parents). This produces **interventional** counterfactuals: "what would a randomly drawn female individual with these parent features look like?" — not the strict individual-level counterfactual "what would *this specific person* look like if they had been female?"

For individual-level CFs (Kusner et al. NeurIPS 2017), abduction must infer each individual's latent background variables before forward-simulating under the intervention. This requires `dowhy.gcm.counterfactual_samples`. Implementing this is a Phase 0 extension (currently marked as a future experiment).

### Why train/test discipline matters

Mechanisms, manifold filter, and classifier are fitted on the **training set only**. Test rows are never seen during fitting. This is essential: manifold membership must be judged relative to the same distribution the model learned, not the evaluation data.

### Paper target

FAccT 2027 or AIES 2027. The three datasets across two distinct failure modes (hard-rule false negatives, soft-dependency false negatives) give sufficient empirical breadth for a systems/methods paper.
