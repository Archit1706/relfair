"""
PDF report generation — Jinja2 template → WeasyPrint → bytes.

Design: the template lives in templates/ll144.html.j2 and uses the FairLens
design tokens. WeasyPrint renders it to a byte stream suitable for writing
to a file or streaming from an API response.

WeasyPrint is an optional dependency (``pip install relfair[report]``).
This module degrades gracefully: if WeasyPrint is not installed, ``render_pdf``
raises ``ImportError`` with a clear install message.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from relfair.metrics.ll144 import LL144Result

_TEMPLATE_DIR = Path(__file__).parent / "templates"
_TEMPLATE_NAME = "ll144.html.j2"


def render_html(result: LL144Result) -> str:
    """
    Render the LL 144 result to an HTML string using the Jinja2 template.

    Returns the rendered HTML (UTF-8 string). Useful for debugging,
    email delivery, or serving as a web page.
    """
    try:
        from jinja2 import Environment, FileSystemLoader, select_autoescape
    except ImportError:
        raise ImportError(
            "Jinja2 is required for report generation. "
            "Install it with: pip install relfair[report]"
        )

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
    )
    # Custom filter for Jinja2
    env.filters["unique"] = lambda seq: list(dict.fromkeys(seq))

    template = env.get_template(_TEMPLATE_NAME)
    return template.render(result=result, meta=result.meta)


def render_pdf(result: LL144Result) -> bytes:
    """
    Render the LL 144 result to a PDF byte string.

    Requires WeasyPrint + GTK3 system libraries.
    - Linux/macOS: ``pip install weasyprint`` (GTK3 usually pre-installed)
    - Windows: install GTK3 runtime from https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer
      then ``pip install weasyprint``

    Raises:
        ImportError: if WeasyPrint is not installed or GTK3 is missing.
    """
    try:
        from weasyprint import HTML  # noqa: F401 — triggers GTK load at import time
    except (ImportError, OSError) as exc:
        raise ImportError(
            "WeasyPrint could not load required system libraries.\n"
            "  Linux/macOS: pip install relfair[report]\n"
            "  Windows:     install GTK3 runtime first — see\n"
            "               https://doc.courtbouillon.org/weasyprint/stable/first_steps.html\n"
            f"  Original error: {exc}"
        ) from exc

    html_str = render_html(result)
    pdf_bytes = HTML(string=html_str, base_url=str(_TEMPLATE_DIR)).write_pdf()
    return pdf_bytes


def write_pdf(result: LL144Result, path: str | Path) -> None:
    """Render and write the PDF report to *path*."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_pdf(result))


def write_html(result: LL144Result, path: str | Path) -> None:
    """Render and write the HTML report to *path* (for preview / debugging)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(result), encoding="utf-8")
