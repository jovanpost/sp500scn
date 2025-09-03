import pandas as pd
from pandas.io.formats.style import Styler

# JavaScript snippet enabling row selection. Injected only once per session.
ROW_SELECT_JS = """
<script id="row-select-js">
document.addEventListener('click', function(e) {
  const row = e.target.closest('tr');
  if (!row) return;
  const table = row.closest('table');
  if (!table) return;
  table.querySelectorAll('tr.selected').forEach(r => r.classList.remove('selected'));
  row.classList.add('selected');
});
</script>
<style id="row-select-style">
table tr.selected {background-color: var(--table-hover); color: var(--table-hover-text);}
</style>
"""

# Internal flag so the selection script is only emitted once.
_row_select_script_injected = False


def inject_row_select_js(table_html: str) -> str:
    """Prepend row-selection script once, returning resulting HTML."""
    global _row_select_script_injected
    if not _row_select_script_injected and "row-select-js" not in table_html:
        _row_select_script_injected = True
        return ROW_SELECT_JS + table_html
    return table_html


def _style_negatives(df: pd.DataFrame) -> Styler:
    """Return a Styler adding class "neg" or "pos" to numeric cells."""
    classes = pd.DataFrame("", index=df.index, columns=df.columns)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        classes.loc[df[col] < 0, col] = "neg"
        classes.loc[df[col] > 0, col] = "pos"
    return df.style.set_td_classes(classes)

