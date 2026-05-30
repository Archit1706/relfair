# `relfair.report`

## Functions

```python
from relfair.report import render_html, render_pdf, write_pdf, to_dict, to_json, write_json
```

### HTML / PDF

```python
html: str = render_html(result)
# Renders the Jinja2 template to an HTML string. No native deps.

pdf_bytes: bytes = render_pdf(result)
# Renders to PDF bytes via WeasyPrint. Requires GTK3 on Windows.

write_pdf(result, "report.pdf")
# Convenience wrapper: render_pdf + write to path.
```

### JSON

```python
d: dict = to_dict(result)
# Converts LL144Result to a plain Python dict (JSON-serialisable).

json_str: str = to_json(result, indent=2)
# Converts to a JSON string.

write_json(result, "report.json")
# Convenience wrapper: to_json + write to path.
```

## PDF requirements

WeasyPrint requires the GTK3 runtime:

- **Linux / Docker**: install `libpango-1.0-0`, `libcairo2`, `libgdk-pixbuf2.0-0` (or use the `weasyprint` Docker image).
- **macOS**: `brew install pango cairo gdk-pixbuf libffi`.
- **Windows**: [GTK3 installer](https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer).

If GTK3 is unavailable, use `render_html()` and open in a browser, or use `--html` in the CLI.
