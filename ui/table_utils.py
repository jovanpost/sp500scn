import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler


ROW_SELECT_JS = """
<script>
(function () {
  const target = window.parent.document;
  target.addEventListener('click', (e) => {
    const row = e.target.closest('div.table-wrapper tbody tr');
    if (!row) return;
    const rows = row.parentElement.querySelectorAll('tr');
    rows.forEach(r => r.classList.remove('selected'));
    row.classList.add('selected');
  });
})();
</script>
<style>
  .table-wrapper tr.selected {
    background-color: var(--table-hover);
    color: var(--table-hover-text);
  }
</style>
"""


def _style_negatives(df: pd.DataFrame) -> Styler:
    """Return a Styler adding class "neg" or "pos" to numeric cells."""
    classes = pd.DataFrame("", index=df.index, columns=df.columns)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        classes.loc[df[col] < 0, col] = "neg"
        classes.loc[df[col] > 0, col] = "pos"
    return df.style.set_td_classes(classes)


def inject_row_select_js(html: str) -> str:
    """Prepend row-select JavaScript once per Streamlit session."""
    injected = st.session_state.setdefault("row_select_js_injected", False)
    if not injected:
        st.session_state["row_select_js_injected"] = True
        return ROW_SELECT_JS + html
    return html

