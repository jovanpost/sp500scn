import pandas as pd
from pandas.io.formats.style import Styler
import streamlit as st


_ROW_SELECT_JS_ID = "row-select-js"
_ROW_SELECT_JS_FLAG = "_row_select_js_injected"
_row_select_js_injected = False

_ROW_SELECT_JS = f"""
<script id="{_ROW_SELECT_JS_ID}">
document.addEventListener('DOMContentLoaded', function () {{
  const rows = window.parent.document.querySelectorAll('tbody tr');
  rows.forEach(row => {{
    row.addEventListener('click', () => {{
      rows.forEach(r => r.classList.remove('selected'));
      row.classList.add('selected');
    }});
  }});
}});
</script>
"""


def inject_row_select_js(table_html: str) -> str:
    """Ensure the row select JS snippet is injected only once.

    If the marker script is already present in ``table_html`` the session and
    module level flags are still set so that subsequent calls will not inject
    the snippet again.
    """

    global _row_select_js_injected
    if _ROW_SELECT_JS_ID in table_html:
        st.session_state[_ROW_SELECT_JS_FLAG] = True
        _row_select_js_injected = True
        return table_html

    if _row_select_js_injected or st.session_state.get(_ROW_SELECT_JS_FLAG):
        return table_html

    st.session_state[_ROW_SELECT_JS_FLAG] = True
    _row_select_js_injected = True
    return f"{table_html}{_ROW_SELECT_JS}"


def _style_negatives(df: pd.DataFrame) -> Styler:
    """Return a Styler adding class "neg" or "pos" to numeric cells."""
    classes = pd.DataFrame("", index=df.index, columns=df.columns)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        classes.loc[df[col] < 0, col] = "neg"
        classes.loc[df[col] > 0, col] = "pos"
    return df.style.set_td_classes(classes)

