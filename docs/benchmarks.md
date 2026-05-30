# Benchmarks

All three experiments are in the `experiments/` directory and are fully reproducible.

## Summary

| Dataset | Naive flip rate | Rel-aware flip rate | Detection lift |
|---|---:|---:|---:|
| Adult — Husband rows (n=3,682) | 7.0% | 24.3% | **+17.2 pp** |
| ACS Income CA (n=5,241) | 7.5% | 34.5% | **+27.0 pp (4.6×)** |
| German Credit (n=144) | 4.9% | 13.2% | **+8.3 pp** |

## Adult / Census Income

**Dataset:** UCI Adult Census Income (48,842 rows). Binary outcome: income > $50k.

**Key constraint:** `relationship = Husband` is almost perfectly correlated with `sex = Male` in the training data. A naive flip `Male → Female` leaves `Husband` unchanged — the model reads the row as anomalous and its prediction is unreliable.

**Hard rules applied:**
```python
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Wife",      from_val="Husband")
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Own-child", from_val="Own-child")
G.add_hard_rule("relationship", when={"sex": "Female"}, value="Not-in-family", from_val="Not-in-family")
```

**Result:** +17.2 pp on Husband rows (the subgroup where the proxy fires). 100% of naive counterfactuals for Husband rows violate the hard rule — `detect_constraint_violations` catches them all; IsolationForest catches ~7.5%.

**Reproduce:**
```bash
cd experiments/adult
python run.py
```

---

## ACS Income CA (folktables)

**Dataset:** American Community Survey 2018, California, income task (n=5,241 test rows).

**Key constraint:** ACS encodes sex, occupation, marital status, and race as float codes. `SEX=1.0` (Male) is strongly associated with certain occupation codes (`OCCP`). Naive flip misses the occupation co-movement entirely.

**Setup:** `cat_cols=["SEX", "MAR", "OCCP", "RAC1P"]` declared explicitly so `LearnedMechanisms` treats float-coded categoricals correctly.

**Result:** +27.0 pp (4.6× detection lift) — the largest lift across our three datasets, driven by the breadth of occupation proxy effects.

**Reproduce:**
```bash
cd experiments/folktables
python run.py --n-train 3000 --n-test 1000   # quick
python run.py                                 # full (downloads ACS data)
```

---

## German Credit

**Dataset:** Statlog German Credit (1,000 rows). Binary outcome: credit risk (good/bad).

**Key constraint:** `personal_status` encodes both sex and marital status jointly (e.g. `male single`, `female div/dep/mar`). Naive flip leaves this compound attribute unchanged.

**Result:** +8.3 pp. Smaller lift than Adult/ACS because German Credit has fewer rows, noisier labels, and the proxy is partially absorbed by the model's other features.

**Reproduce:**
```bash
cd experiments/german_credit
python run.py
```

---

## Constraint violation detector vs. IsolationForest

A key secondary finding: `detect_constraint_violations()` is **definitionally correct** for hard-rule incoherences. A row with `sex=Female, relationship=Husband` is by definition off-manifold — it never occurs in training data. No probabilistic method can beat 100% recall here.

IsolationForest scores ~7.5% of rows as anomalous regardless of whether a hard-rule violation is present. It cannot localise specific attribute-pair contradictions. Use `ManifoldFilter` for soft distributional drift only.
