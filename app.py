# app.py
import os
import streamlit as st

from ui.layout import setup_page, render_header
from ui.price_filter import initialize_price_filter
from ui.nav import TABS

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

# Initialize page and layout/CSS
setup_page()
initialize_price_filter()

# Brand/header
render_header()

# Build tabs exclusively from the registry (single source of truth)
tab_containers = st.tabs([label for (label, _route, _render) in TABS])

for container, (label, _route, render_fn) in zip(tab_containers, TABS):
    with container:
        if callable(render_fn):
            render_fn()
        else:
            st.warning(f"{label} is unavailable (import failed).")
