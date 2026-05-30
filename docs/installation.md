# Installation

## Requirements

- Python 3.10, 3.11, or 3.12
- pip ≥ 23

## Install options

```bash
# Core library — counterfactual engine + LL 144 metrics
pip install relfair

# Add the CLI and PDF report generation
pip install "relfair[cli,report]"

# Add benchmark experiment dependencies
pip install "relfair[experiments]"

# Everything
pip install "relfair[all]"
```

## PDF reports on Windows

WeasyPrint (used by `relfair[report]`) requires the GTK3 runtime on Windows.
Follow the [WeasyPrint installation guide](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html).

The `--html` flag on the CLI generates a standalone HTML report with no native dependencies:

```bash
relfair audit predictions.csv --outcome hired --sex sex --race race --html report.html
```

## Development install

```bash
git clone https://github.com/Archit1706/relfair
cd relfair
pip install -e ".[dev]"

# Run the test suite
python -m pytest tests/

# Lint + type-check
ruff check relfair/ tests/
mypy relfair/
```
