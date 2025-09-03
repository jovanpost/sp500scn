import pandas as pd
from pandas.io.formats.style import Styler
from typing import Union

ROW_SELECT_JS = """
<script id="row-select-js">
(function() {
    if (window.__rowNavInit) return;
    window.__rowNavInit = true;
    document.addEventListener('click', function(e) {
        const row = e.target.closest('tr[data-href]');
        if (row && row.dataset.href) {
            window.location.href = row.dataset.href;
        }
    });
})();
</script>
"""

_script_emitted = False

def row_select_script() -> str:
    """Return row-selection script, ensuring it is emitted only once."""
    global _script_emitted
    if _script_emitted:
        return ""
    _script_emitted = True
    return ROW_SELECT_JS

def render_table(df: Union[pd.DataFrame, Styler], *, include_script: bool = True) -> str:
    """Return HTML for ``df`` with optional row-selection script.

    The script is included only the first time this function is called
    (or when ``include_script`` is False).
    """
    table_html = df.to_html()
    script = row_select_script() if include_script else ""
    return script + table_html
