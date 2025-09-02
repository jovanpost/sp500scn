import streamlit as st
from streamlit.components.v1 import html

from ui.layout import setup_page, render_header
from ui.scan import render_scanner_tab
from ui.history import render_history_tab
from ui.debugger import render_debugger_tab

# Initialize page and global layout/CSS
setup_page()

# Detect viewport size
viewport = html(
    """
    <script>
    const send = () => {
        const dims = {width: window.innerWidth, height: window.innerHeight};
        Streamlit.setComponentValue(dims);
    };
    window.addEventListener('load', send);
    window.addEventListener('resize', send);
    send();
    </script>
    """,
    height=0,
    width=0,
)
if viewport and isinstance(viewport, dict):
    st.session_state["viewport"] = viewport

# ---- Brand header ----
render_header()

width = st.session_state.get("viewport", {}).get("width", 1000)

if width < 400:
    nav = st.selectbox(
        "Navigation",
        ["📈 Scanner", "📊 History & Outcomes", "🐞 Debugger"],
    )
    if nav.startswith("📈"):
        render_scanner_tab()
    elif nav.startswith("📊"):
        render_history_tab()
    else:
        render_debugger_tab()
else:
    # Create tabs once with unique variable names
    tab_scanner, tab_history, tab_debug = st.tabs(
        ["📈 Scanner", "📊 History & Outcomes", "🐞 Debugger"]
    )

    with tab_scanner:
        render_scanner_tab()

    with tab_history:
        render_history_tab()

    with tab_debug:
        render_debugger_tab()
