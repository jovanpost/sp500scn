import importlib.util
from pathlib import Path
import streamlit as st
from ui.layout import setup_page, render_header

# Initialize page and global layout/CSS
setup_page()

# ---- Brand header ----
render_header()

# Create tabs once with unique variable names
tab_gap, tab_history, tab_lake, tab_debug = st.tabs(
    ["⚡ Gap Scanner", "📈 History & Outcomes", "💧 Data Lake (Phase 1)", "🐞 Debugger"]
)

with tab_gap:
    spec = importlib.util.spec_from_file_location(
        "ui.pages.yday_vol_signal_open", Path("ui/pages/45_YdayVolSignal_Open.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.page()

with tab_history:
    from ui.history import render_history_tab

    render_history_tab()

with tab_lake:
    spec = importlib.util.spec_from_file_location(
        "ui.pages.data_lake_phase1", Path("ui/pages/90_Data_Lake_Phase1.py")
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    mod.render_data_lake_tab()

with tab_debug:
    from ui.debugger import render_debugger_tab

    render_debugger_tab()
