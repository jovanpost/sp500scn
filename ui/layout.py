import streamlit as st
from pathlib import Path
from streamlit import components


def _inject_sticky_helper():
    helper_js = Path(__file__).with_name("sticky_df_helper.js").read_text()
    components.v1.html(f"<script>{helper_js}</script>", height=0, width=0, scrolling=False)


def setup_page(*, table_hover: str = "#2563eb", table_hover_text: str = "#ffffff"):
    st.set_page_config(
        page_title="Edge500",  # Title shown in browser tab
        page_icon="logo.png",  # Favicon (logo.png in repo root)
        layout="wide",
    )

    primary = "#ff6b6b"
    secondary = "#3399ff"
    background = "#000000"
    text = "#f2f2f2"

    # Import Google font
    st.markdown(
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap' rel='stylesheet'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <style>
        :root {{
            --color-primary: {primary};
            --color-secondary: {secondary};
            --bg-color: {background};
            --text-color: {text};
            --padding: 1rem;
            --font-size-base: 16px;
            --col-width: 33%;
        }}

        .dark-table,
        div[data-testid="stDataFrame"] {{
            --table-bg: #1f2937;
            --table-header-bg: #374151;
            --table-row-alt: #1e293b;
            --table-hover: {table_hover};
            --table-hover-text: {table_hover_text};
            --table-text: #e5e7eb;
            --table-header-text: #f9fafb;
            --table-border: #4b5563;
            --table-pos: #22c55e;
            --table-neg: #ef4444;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background: var(--bg-color);
            color: var(--text-color);
        }}

        /* --- Buttons / general --- */
        div.stButton > button:first-child {{
            background-color: var(--color-primary) !important;
            color: var(--color-secondary) !important;
            font-weight: 700 !important;
        }}

        /* --- WHY BUY text block --- */
        .whybuy {{ font-size: calc(var(--font-size-base) - 2px); line-height: 1.55; }}

        /* --- Debugger layout (HTML) --- */
        .dbg-wrap {{ max-width: 1100px; margin-top: 8px; }}
        .dbg-title {{ font-size: 28px; font-weight: 800; letter-spacing: .2px; margin: 4px 0 12px; }}
        .dbg-badge {{
            display:inline-block; padding: 4px 10px; margin-left: 10px;
            border-radius: 999px; font-size: 13px; font-weight: 700;
            vertical-align: middle;
        }}
        .dbg-badge.fail {{ background:#ffe6e6; color:#b00020; border:1px solid #ffb3b3; }}
        .dbg-badge.pass {{ background:#e7f6ec; color:#0a7a35; border:1px solid #bfe6cc; }}
        .dbg-subtle {{ color:#666; font-size: 14px; margin-bottom: 10px; }}
        .dbg-snapshot {{
            background:#f7f7f9; border-left:4px solid #c7c7d1;
            padding:10px 12px; margin: 14px 0 10px; font-size:15px;
        }}
        .dbg-snap-kv {{ display:inline-block; margin-right: 14px; }}
        .dbg-snap-kv .k {{ color:#666; }}
        .dbg-snap-kv .v {{ font-weight:700; color:#111; }}
        .dbg-json details {{ margin-top: 10px; }}
        .dbg-json summary {{ cursor: pointer; font-weight: 700; }}
        .dbg-json pre {{
            background:#111; color:#f2f2f2; padding:12px; border-radius:8px;
            overflow:auto; font-size:13px; line-height:1.45;
        }}
        .em {{ font-style: italic; }}

        /* --- Hero header --- */
        .hero {{
            background: linear-gradient(135deg, var(--color-primary), var(--color-secondary));
            background-size: 400% 400%;
            animation: gradientShift 8s ease infinite;
            padding: 2rem 1rem;
            text-align: center;
            border-radius: 8px;
            margin: 0.5rem 0 1rem;
        }}
        @keyframes gradientShift {{
            0% {{background-position: 0% 50%;}}
            50% {{background-position: 100% 50%;}}
            100% {{background-position: 0% 50%;}}
        }}

        /* --- Responsive tabs --- */
        .stTabs [role="tablist"] {{
            flex-wrap: wrap;
            gap: 0.25rem;
        }}
        .stTabs [role="tab"] {{
            padding: 0.5rem 1rem;
        }}
        @media (max-width: 600px) {{
            .stTabs [role="tablist"] {{
                overflow-x: auto;
                flex-wrap: nowrap;
            }}
            .stTabs [role="tab"] {{
                flex: 1 0 auto;
                white-space: nowrap;
            }}
        }}

        @media (max-width: 600px) {{
            :root {{
                --padding: 0.5rem;
                --font-size-base: 14px;
                --col-width: 100%;
            }}
            .dbg-title {{ font-size: 22px; }}
        }}

        @media (min-width: 600px) and (max-width: 900px) {{
            :root {{
                --padding: 0.75rem;
                --font-size-base: 15px;
                --col-width: 50%;
            }}
            .dbg-title {{ font-size: 24px; }}
        }}

        @media (min-width: 900px) {{
            :root {{
                --padding: 1rem;
                --font-size-base: 16px;
                --col-width: 33%;
            }}
        }}

        .block-container {{
            padding-left: var(--padding);
            padding-right: var(--padding);
        }}

        /* --- DataFrame tables --- */
        table.dark-table {{
            background-color: var(--table-bg);
            border: 1px solid var(--table-border);
            border-radius: 8px;
            border-collapse: separate;
            border-spacing: 0;
            width: max-content;
        }}
        /* Header */
        table.dark-table thead th {{
            position: sticky;
            top: 0;
            z-index: 3;
            background-color: var(--table-header-bg);
        }}
        table.dark-table thead th:first-child {{
            left: 0;
            z-index: 5;
        }}
        /* Rows */
        table.dark-table tbody tr {{
            background-color: var(--table-bg);
            color: var(--table-text);
        }}
        table.dark-table tbody tr:nth-child(even) {{
            background-color: var(--table-row-alt);
        }}
        /* Hover effect */
        table.dark-table tbody tr:hover {{
            background-color: var(--table-hover);
            color: var(--table-hover-text);
        }}
        /* Borders between cells */
        table.dark-table td,
        table.dark-table th {{
            border-bottom: 1px solid var(--table-border);
            padding: 8px;
        }}
        /* Positive / Negative number coloring */
        table.dark-table td.pos {{
            color: var(--table-pos) !important;
            font-weight: 600;
        }}
        table.dark-table td.neg {{
            color: var(--table-neg) !important;
            font-weight: 600;
        }}

        /* Streamlit DataFrame styling and sticky headers */
        /* Make the DataFrame root the scroll container */
        div[data-testid="stDataFrame"] {{
            position: relative;
            overflow: auto !important; /* override Streamlit’s overflow: hidden */
            border-radius: 10px;
        }}

        /* Ensure table rendering works well with sticky headers */
        div[data-testid="stDataFrame"] table {{
            border-collapse: separate;
            border-spacing: 0;
        }}

        /* Sticky header cells (native <th> and grid role headers) */
        div[data-testid="stDataFrame"] thead th,
        div[data-testid="stDataFrame"] [role="columnheader"] {{
            position: sticky;
            top: 0;
            z-index: 5; /* above rows and hover backgrounds */
            background: var(--table-header-bg);
            color: var(--table-header-text);
            box-shadow: inset 0 -1px 0 var(--table-border); /* subtle divider line */
        }}

        /* Row striping and base background */
        div[data-testid="stDataFrame"].sticky-scroll tbody tr td,
        div[data-testid="stDataFrame"] .sticky-scroll tbody tr td {{
            background: var(--table-bg);
        }}
        div[data-testid="stDataFrame"].sticky-scroll tbody tr:nth-child(even) td,
        div[data-testid="stDataFrame"] .sticky-scroll tbody tr:nth-child(even) td {{
            background: var(--table-row-alt);
        }}

        /* Hover highlight that does not cover the header */
        div[data-testid="stDataFrame"].sticky-scroll tbody tr:hover td,
        div[data-testid="stDataFrame"] .sticky-scroll tbody tr:hover td {{
            background: var(--table-hover);
            color: var(--table-hover-text);
        }}

        /* Optional: subtle border around the scrolling frame so it reads as a card */
        div[data-testid="stDataFrame"].sticky-scroll,
        div[data-testid="stDataFrame"] .sticky-scroll {{
            border: 1px solid var(--table-border);
            border-radius: 10px;
        }}

        /* Scrollable wrapper for custom HTML tables */
        .table-wrapper {{
            position: relative;
            overflow-x: auto;
            overflow-y: auto;
        }}
        .table-wrapper table {{
            width: max-content;
        }}
        .table-wrapper thead th {{
            position: sticky;
            top: 0;
            z-index: 2;
            background-color: var(--table-header-bg);
        }}
        .table-wrapper tbody tr:hover {{
            background-color: var(--table-hover);
            color: var(--table-hover-text);
        }}
        .table-wrapper tbody tr:hover td:first-child {{
            background-color: var(--table-hover);
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _inject_sticky_helper()

    st.markdown(
        """
        <script>
        document.addEventListener('keydown', function(event) {
            const active = document.activeElement;
            if (active && active.classList.contains('table-wrapper')) {
                const scrollAmount = 40;
                if (event.key === 'ArrowRight') {
                    active.scrollLeft += scrollAmount;
                    event.preventDefault();
                } else if (event.key === 'ArrowLeft') {
                    active.scrollLeft -= scrollAmount;
                    event.preventDefault();
                }
            }
        });
        </script>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    """Render animated hero header with logo."""
    header = st.container()
    with header:
        st.markdown('<div class="hero">', unsafe_allow_html=True)
        st.image("logo.png", width=140)
        st.markdown("</div>", unsafe_allow_html=True)
    st.divider()
