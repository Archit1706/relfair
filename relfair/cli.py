"""
relfair CLI — entry point for the LL 144 audit command.

Usage:
    relfair audit predictions.csv \\
        --outcome hired \\
        --sex sex \\
        --race race \\
        --pdf report.pdf \\
        --json report.json \\
        --employer "Acme Corp" \\
        --aedt "Resume Ranker v4.2" \\
        --auditor "Archit Rathod" \\
        --cover-period "Jan 1 2025 - Dec 31 2025"

Install:
    pip install relfair[cli,report]
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import click


# ── helpers ──────────────────────────────────────────────────────────────────

def _rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def _print_group_table(
    stats,
    title: str,
    use_rich: bool,
) -> None:
    if use_rich:
        from rich.console import Console
        from rich.table import Table
        console = Console()
        tbl = Table(title=title, show_header=True, header_style="bold white on #14130F")
        tbl.add_column("Group",         style="bold")
        tbl.add_column("n",             justify="right")
        tbl.add_column("Selected",      justify="right")
        tbl.add_column("Rate",          justify="right")
        tbl.add_column("Impact ratio",  justify="right")
        tbl.add_column("95% CI",        justify="right")
        tbl.add_column("4/5 Rule",      justify="center")
        for s in stats:
            flag_str = "[bold #B0411B]FLAG[/]" if s.four_fifths_flag else "[#2D5F3F]PASS[/]"
            suffix   = " ⚠" if s.small_sample else ""
            tbl.add_row(
                s.group + (" (ref)" if s.is_reference else ""),
                f"{s.n:,}{suffix}",
                f"{s.selected:,}",
                f"{s.rate*100:.1f}%",
                f"{s.ratio:.3f}",
                f"[{s.ratio_ci_lo:.3f}, {s.ratio_ci_hi:.3f}]",
                flag_str,
            )
        console.print(tbl)
    else:
        click.echo(f"\n{title}")
        click.echo("-" * 80)
        click.echo(f"{'Group':<35} {'n':>7} {'Rate':>7} {'Ratio':>7}  {'4/5':>5}")
        click.echo("-" * 80)
        for s in stats:
            flag = "FLAG" if s.four_fifths_flag else "PASS"
            suffix = "⚠" if s.small_sample else " "
            click.echo(
                f"{s.group:<35} {s.n:>7,}{suffix} "
                f"{s.rate*100:>6.1f}% {s.ratio:>7.3f}  {flag:>5}"
            )


# ── main command ──────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="relfair")
def cli() -> None:
    """relfair — relationship-aware counterfactual fairness testing."""


@cli.command("audit")
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option("--outcome",       required=True,  help="Column name for outcome (binary 0/1 or score)")
@click.option("--outcome-type",  default="binary", type=click.Choice(["binary", "score"]), show_default=True)
@click.option("--sex",           required=True,  help="Column name for sex / gender")
@click.option("--race",          required=True,  help="Column name for race / ethnicity")
@click.option("--pdf",           default=None,   help="Output path for the PDF report (requires WeasyPrint)")
@click.option("--json",    "json_out", default=None, help="Output path for the machine-readable JSON")
@click.option("--html",          default=None,   help="Output path for the HTML report (preview)")
@click.option("--n-boot",        default=2000,   show_default=True, help="Bootstrap resamples for CIs")
@click.option("--seed",          default=0,      show_default=True, help="RNG seed")
@click.option("--employer",      default="",     help="Employer name (appears in report)")
@click.option("--aedt",          default="",     help="AEDT name and version")
@click.option("--auditor",       default="",     help="Auditor name (appears in signature block)")
@click.option("--cover-period",  default="",     help="Audit coverage period, e.g. 'Jan 1 2025 - Dec 31 2025'")
@click.option("--quiet",         is_flag=True,   help="Suppress console output")
def audit(
    input_file: str,
    outcome: str,
    outcome_type: str,
    sex: str,
    race: str,
    pdf: str | None,
    json_out: str | None,
    html: str | None,
    n_boot: int,
    seed: int,
    employer: str,
    aedt: str,
    auditor: str,
    cover_period: str,
    quiet: bool,
) -> None:
    """
    Run a NYC LL 144 bias audit on INPUT_FILE.

    INPUT_FILE must be a CSV with at least the outcome column, a sex column,
    and a race/ethnicity column.  Additional columns are ignored.

    At least one of --pdf, --json, or --html must be specified, or results
    are printed to the console only.

    Example:

        relfair audit predictions.csv \\
            --outcome hired --sex sex --race race \\
            --pdf report.pdf --json report.json \\
            --employer "Acme Corp" --auditor "Archit Rathod"
    """
    import pandas as pd

    use_rich = _rich_available() and not quiet

    if use_rich:
        from rich.console import Console
        console = Console()
        console.rule("[bold]relfair · NYC LL 144 Bias Audit[/]")
    elif not quiet:
        click.echo("=" * 60)
        click.echo("  relfair — NYC LL 144 Bias Audit")
        click.echo("=" * 60)

    # ── load data ────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        click.echo(f"ERROR: could not read {input_file}: {e}", err=True)
        sys.exit(1)

    for col in [outcome, sex, race]:
        if col not in df.columns:
            click.echo(
                f"ERROR: column '{col}' not found in {input_file}.\n"
                f"Available columns: {list(df.columns)}",
                err=True,
            )
            sys.exit(1)

    if not quiet:
        msg = f"  Loaded {len(df):,} rows from {Path(input_file).name}"
        click.echo(msg) if not use_rich else console.print(msg)

    # ── compute metrics ──────────────────────────────────────────────────
    from relfair.metrics import compute_ll144_metrics

    if not quiet:
        msg = f"  Computing LL 144 metrics  (bootstrap n={n_boot:,}, seed={seed})..."
        click.echo(msg) if not use_rich else console.print(msg)

    meta = {
        "employer":       employer,
        "aedt_name":      aedt,
        "auditor":        auditor,
        "cover_period":   cover_period,
        "audit_date":     date.today().isoformat(),
        "n_boot":         n_boot,
        "bootstrap_seed": seed,
        "relfair_version": _version(),
    }

    result = compute_ll144_metrics(
        df,
        outcome_col=outcome,
        outcome_type=outcome_type,
        sex_col=sex,
        race_col=race,
        n_boot=n_boot,
        bootstrap_seed=seed,
        meta=meta,
    )

    # ── console output ───────────────────────────────────────────────────
    if not quiet:
        if use_rich:
            console.print(
                f"\n  [bold]Applicants:[/] {result.n_total:,}  "
                f"[bold]Selected:[/] {result.n_selected:,}  "
                f"[bold]Overall rate:[/] {result.overall_rate*100:.1f}%"
            )
        else:
            click.echo(
                f"\n  Applicants: {result.n_total:,}  "
                f"Selected: {result.n_selected:,}  "
                f"Overall rate: {result.overall_rate*100:.1f}%"
            )

        _print_group_table(result.by_sex,  "Selection rates by sex",           use_rich)
        _print_group_table(result.by_race, "Selection rates by race/ethnicity", use_rich)

        # Intersectional summary — just flag count
        n_inter_flags = sum(1 for c in result.intersectional if c.four_fifths_flag)
        total_cells   = len(result.intersectional)
        msg = f"\n  Intersectional: {n_inter_flags}/{total_cells} cells flagged"
        if use_rich:
            colour = "#B0411B" if n_inter_flags else "#2D5F3F"
            console.print(f"[bold {colour}]{msg}[/]")
        else:
            click.echo(msg)

    # ── write outputs ────────────────────────────────────────────────────
    if json_out:
        from relfair.report import write_json
        write_json(result, json_out)
        if not quiet:
            click.echo(f"\n  JSON  -> {json_out}")

    if html:
        from relfair.report import write_html
        write_html(result, html)
        if not quiet:
            click.echo(f"  HTML  -> {html}")

    if pdf:
        try:
            from relfair.report import write_pdf
            if not quiet:
                click.echo("  Generating PDF (WeasyPrint)...")
            write_pdf(result, pdf)
            if not quiet:
                click.echo(f"  PDF   -> {pdf}")
        except ImportError as exc:
            click.echo(f"WARNING: PDF skipped — {exc}", err=True)
            click.echo(
                "  Tip: generate --html instead (no native deps required).",
                err=True,
            )

    if not quiet:
        click.echo()
        if use_rich:
            console.rule("[bold]Done[/]")
        else:
            click.echo("Done.")


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("relfair")
    except Exception:
        return "0.1.0"


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
