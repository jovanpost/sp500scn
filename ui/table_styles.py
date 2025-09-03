import pandas as pd
from pandas.io.formats.style import Styler


def _apply_dark_theme(df: pd.DataFrame | Styler) -> Styler:
    base = df.style if isinstance(df, pd.DataFrame) else df
    return base.set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "var(--table-header-bg)"),
                ("color", "var(--table-header-text)"),
                ("border", "1px solid var(--table-border)"),
            ],
        },
        {
            "selector": "td",
            "props": [
                ("background-color", "var(--table-bg)"),
                ("color", "var(--table-text)"),
                ("border", "1px solid var(--table-border)"),
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
    ])

