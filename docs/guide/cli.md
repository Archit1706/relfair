# CLI Reference

## Install

```bash
pip install "relfair[cli,report]"
```

## `relfair audit`

Run an LL 144 bias audit on a predictions CSV.

```
Usage: relfair audit [OPTIONS] INPUT_FILE

Arguments:
  INPUT_FILE  Path to the CSV file containing predictions.

Options:
  --outcome TEXT           Column name for outcome (binary 0/1 or score)  [required]
  --outcome-type TEXT      binary or score  [default: binary]
  --sex TEXT               Column name for sex / gender  [required]
  --race TEXT              Column name for race / ethnicity  [required]
  --pdf PATH               Output path for the PDF report
  --json PATH              Output path for the machine-readable JSON
  --html PATH              Output path for the HTML report (no native deps)
  --n-boot INTEGER         Bootstrap resamples for CIs  [default: 2000]
  --seed INTEGER           RNG seed  [default: 0]
  --employer TEXT          Employer name (appears in report header)
  --aedt TEXT              AEDT name and version
  --auditor TEXT           Auditor name (appears in signature block)
  --cover-period TEXT      Audit coverage period, e.g. "Jan 1 2025 - Dec 31 2025"
  --quiet                  Suppress console output
  --help                   Show this message and exit.
```

## Example

```bash
relfair audit predictions.csv \
  --outcome   hired \
  --sex       sex \
  --race      race \
  --pdf       report.pdf \
  --json      report.json \
  --employer  "Northbound Talent" \
  --aedt      "Resume Ranker v4.2" \
  --auditor   "Archit Rathod" \
  --cover-period "Jan 1 2025 – Dec 31 2025"
```

## Input CSV format

The CSV must have at least:

| Column | Values |
|---|---|
| outcome column | `0` / `1` (binary) or float (score) |
| sex column | Any string labels, e.g. `Male`, `Female` |
| race column | Any string labels, e.g. `White`, `Black`, `Hispanic`, `Asian` |

Extra columns are ignored. The tool auto-detects the reference group (highest selection rate) unless you pass `--reference-sex` / `--reference-race` in the Python API.

## Console output

```
                     LL 144 Audit — Acme Corp / Resume Ranker v4.2
┌──────────────────────┬───────┬──────────┬────────┬───────────────────┬───────┐
│ Group                │   n   │ Selected │  Rate  │ Ratio (95% CI)    │ Flag  │
├──────────────────────┼───────┼──────────┼────────┼───────────────────┼───────┤
│ Sex: Male (ref)      │  4821 │    1203  │ 24.9%  │ 1.000             │       │
│ Sex: Female          │  3201 │    510   │ 15.9%  │ 0.640 [0.58-0.70] │  ⚑   │
├──────────────────────┼───────┼──────────┼────────┼───────────────────┼───────┤
│ Race: White (ref)    │  3912 │    978   │ 25.0%  │ 1.000             │       │
│ Race: Black          │  1204 │    193   │ 16.0%  │ 0.641 [0.56-0.73] │  ⚑   │
│ Race: Hispanic       │   987 │    210   │ 21.3%  │ 0.850 [0.75-0.96] │       │
│ Race: Asian          │  1919 │    432   │ 22.5%  │ 0.900 [0.82-0.98] │       │
└──────────────────────┴───────┴──────────┴────────┴───────────────────┴───────┘
⚑ = four-fifths flag (ratio < 0.80)
```
