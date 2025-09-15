# Avoid importing heavy modules at import-time; pages import what they need locally.
from .layout import setup_page, render_header
from .history import render_history_tab

# ---------------------------------------------------------------------------
# Minimal HTML table parser for environments without lxml/bs4. Pandas requires
# one of these optional deps for ``read_html`` which our tests rely on. If the
# import fails, provide a very small fallback that handles the simple tables we
# emit (header row + body rows).
import pandas as pd
from html.parser import HTMLParser


def _fallback_read_html(html, *args, **kwargs):
    if hasattr(html, "read"):
        html = html.read()
    rows: list[list[str]] = []

    class _Parser(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.in_cell = False
            self.current = ""

        def handle_starttag(self, tag, attrs):
            if tag == "tr":
                self.row: list[str] = []
            if tag in {"td", "th"}:
                self.in_cell = True
                self.current = ""

        def handle_endtag(self, tag):
            if tag in {"td", "th"} and self.in_cell:
                self.row.append(self.current.strip())
                self.in_cell = False
            if tag == "tr" and getattr(self, "row", None):
                rows.append(self.row)

        def handle_data(self, data):
            if self.in_cell:
                self.current += data

    _Parser().feed(html)
    if not rows:
        return []
    header, *body = rows
    return [pd.DataFrame(body, columns=header)]


try:  # pragma: no cover - exercised in tests
    pd.read_html("<table><tr><td>1</td></tr></table>")
except Exception:  # lxml/bs4/html5lib missing
    pd.read_html = _fallback_read_html  # type: ignore[assignment]

__all__ = ["setup_page", "render_header", "render_history_tab"]
