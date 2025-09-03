import pandas as pd
from pandas.io.formats.style import Styler


# JavaScript snippet that attaches click handlers to dark-themed tables.
# Wrapped in an IIFE so it executes immediately on each Streamlit rerun.
ROW_CLICK_JS = """
<script>
(function() {
    document.querySelectorAll('table.dark-table').forEach(function(table) {
        table.addEventListener('click', function(event) {
            const row = event.target.closest('tr');
            if (!row) return;
            table.querySelectorAll('tr.selected').forEach(function(r) {
                r.classList.remove('selected');
            });
            row.classList.add('selected');
        });
    });
})();
</script>
"""


def _style_negatives(df: pd.DataFrame) -> Styler:
    """Return a Styler adding class "neg" or "pos" to numeric cells."""
    classes = pd.DataFrame("", index=df.index, columns=df.columns)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        classes.loc[df[col] < 0, col] = "neg"
        classes.loc[df[col] > 0, col] = "pos"
    return df.style.set_td_classes(classes)

