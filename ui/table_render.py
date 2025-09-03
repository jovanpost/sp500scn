import pandas as pd
from pandas.io.formats.style import Styler

from .table_utils import _style_negatives


ROW_SELECT_JS = """
<script>
// Allow arrow-key row navigation within table wrappers
(function() {
  function onKey(e) {
    const wrapper = e.target.closest('.table-wrapper');
    if (!wrapper) return;
    const rows = wrapper.querySelectorAll('tbody tr');
    if (!rows.length) return;
    let idx = Array.from(rows).findIndex(r => r.classList.contains('selected'));
    if (e.key === 'ArrowDown') {
      idx = Math.min(idx + 1, rows.length - 1);
    } else if (e.key === 'ArrowUp') {
      idx = Math.max(idx - 1, 0);
    } else {
      return;
    }
    e.preventDefault();
    rows.forEach(r => r.classList.remove('selected'));
    rows[idx].classList.add('selected');
    rows[idx].scrollIntoView({block: 'nearest'});
  }
  document.addEventListener('keydown', onKey, true);
})();
</script>
"""


def _apply_dark_theme(
    df: pd.DataFrame | Styler, colors: dict[str, str] | None = None
) -> Styler:
    """Apply a dark theme with inlined palette and scoped pos/neg styles."""
    palette = {
        "--table-bg": "#1f2937",
        "--table-header-bg": "#374151",
        "--table-row-alt": "#1e293b",
        "--table-hover": "#2563eb",
        "--table-hover-text": "#ffffff",
        "--table-text": "#e5e7eb",
        "--table-header-text": "#f9fafb",
        "--table-border": "#4b5563",
        "--table-pos": "#22c55e",
        "--table-neg": "#ef4444",
    }
    if colors:
        palette.update(colors)

    base = df.style if isinstance(df, pd.DataFrame) else df
    styles = [
        {"selector": ":root", "props": list(palette.items())},
        {
            "selector": "th",
            "props": [
                ("background-color", "var(--table-header-bg)"),
                ("color", "var(--table-header-text)"),
                ("border-bottom", "1px solid var(--table-border)"),
                ("font-weight", "600"),
                ("text-align", "center"),
                ("padding", "8px"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("background-color", "var(--table-bg)"),
                ("color", "var(--table-text)"),
                ("border-bottom", "1px solid var(--table-border)"),
                ("padding", "8px"),
            ],
        },
        {
            "selector": "tbody tr:nth-child(even)",
            "props": [("background-color", "var(--table-row-alt)")],
        },
        {
            "selector": "tbody tr:hover",
            "props": [
                ("background-color", "var(--table-hover)"),
                ("color", "var(--table-hover-text)"),
            ],
        },
        {
            "selector": "table",
            "props": [
                ("border-collapse", "separate"),
                ("border-spacing", "0"),
                ("border-radius", "8px"),
                ("overflow", "hidden"),
                ("width", "max-content"),
                ("white-space", "nowrap"),
            ],
        },
        {
            "selector": "th:first-child",
            "props": [
                ("position", "sticky"),
                ("left", "0"),
                ("z-index", "2"),
                ("background-color", "var(--table-header-bg)"),
            ],
        },
        {
            "selector": "td:first-child",
            "props": [
                ("position", "sticky"),
                ("left", "0"),
                ("background-color", "var(--table-bg)"),
                ("z-index", "1"),
            ],
        },
        {
            "selector": "td.pos",
            "props": [("color", "var(--table-pos)"), ("font-weight", "600")],
        },
        {
            "selector": "td.neg",
            "props": [("color", "var(--table-neg)"), ("font-weight", "600")],
        },
    ]
    return base.set_table_styles(styles)


def render_table(df: pd.DataFrame, *, colors: dict[str, str] | None = None) -> str:
    """Return an HTML table wrapped for Streamlit rendering."""
    html = _apply_dark_theme(_style_negatives(df), colors).to_html()
    return f"<div class='table-wrapper' tabindex='0'>{html}</div>" + ROW_SELECT_JS
