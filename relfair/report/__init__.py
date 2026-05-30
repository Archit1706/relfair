"""
relfair.report — Report generation (HTML, PDF, JSON).

PDF generation requires WeasyPrint: pip install relfair[report]
"""

from .json_export import to_dict, to_json, write_json
from .pdf import render_html, render_pdf, write_html, write_pdf

__all__ = [
    "render_html", "render_pdf", "write_html", "write_pdf",
    "to_dict", "to_json", "write_json",
]
