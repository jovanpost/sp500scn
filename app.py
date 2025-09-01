import streamlit as st

from ui.layout import setup_page, render_header
from ui.scan import render_scanner_tab
from ui.history import render_history_tab
from ui.debugger import render_debugger_tab

# Initialize page and global layout/CSS
setup_page()

# ---- Brand header ----
render_header()

# Create tabs once with unique variable names
tab_scanner, tab_history, tab_debug = st.tabs(
    ["Scanner", "History & Outcomes", "Debugger"]
)

with tab_scanner:
    render_scanner_tab()

with tab_history:
    render_history_tab()

with tab_debug:
    render_debugger_tab()
