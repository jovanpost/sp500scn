import importlib.util
from pathlib import Path
import streamlit as st
from ui.layout import setup_page, render_header
from ui.scan import render_scanner_tab
from ui.history import render_history_tab
from ui.debugger import render_debugger_tab
# Dynamically import the data lake tab
_spec = importlib.util.spec_from_file_location("ui.pages.data_lake_phase1", Path("ui/pages/90_Data_Lake_Phase1.py"))
_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_mod)
render_data_lake_tab = _mod.render_data_lake_tab
# Dynamically import the Gap Scanner page
_spec2 = importlib.util.spec_from_file_location("ui.pages.yday_vol_signal_open", Path("ui/pages/45_YdayVolSignal_Open.py"))
_mod2 = importlib.util.module_from_spec(_spec2)
assert _spec2 and _spec2.loader
_spec2.loader.exec_module(_mod2)
render_gap_scanner = _mod2.page

# Initialize page and global layout/CSS
setup_page()

# ---- Brand header ----
render_header()

# Create tabs once with unique variable names
tab_scanner, tab_gap, tab_history, tab_lake, tab_debug = st.tabs(
    ["🔍 Scanner", "⚡ Gap Scanner", "📈 History & Outcomes", "💧 Data Lake (Phase 1)", "🐞 Debugger"]
)
with tab_scanner:
    render_scanner_tab()
with tab_gap:
    render_gap_scanner()
with tab_history:
    render_history_tab()
with tab_lake:
    render_data_lake_tab()
with tab_debug:
    render_debugger_tab()
