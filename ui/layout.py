import streamlit as st


def setup_page():
    st.set_page_config(
        page_title="Edge500",     # Title shown in browser tab
        page_icon="logo.png",     # Favicon (logo.png in repo root)
        layout="wide",
    )

    st.markdown(
        """
        <style>
        /* --- Buttons / general --- */
        div.stButton > button:first-child {
            background-color: red !important;
            color: white !important;
            font-weight: 700 !important;
        }

        /* --- WHY BUY text block --- */
        .whybuy { font-size: 16px; line-height: 1.55; }

        /* --- Debugger layout (HTML) --- */
        .dbg-wrap { max-width: 1100px; margin-top: 8px; }
        .dbg-title { font-size: 28px; font-weight: 800; letter-spacing: .2px; margin: 4px 0 12px; }
        .dbg-badge {
            display:inline-block; padding: 4px 10px; margin-left: 10px;
            border-radius: 999px; font-size: 13px; font-weight: 700;
            vertical-align: middle;
        }
        .dbg-badge.fail { background:#ffe6e6; color:#b00020; border:1px solid #ffb3b3; }
        .dbg-badge.pass { background:#e7f6ec; color:#0a7a35; border:1px solid #bfe6cc; }
        .dbg-subtle { color:#666; font-size: 14px; margin-bottom: 10px; }
        .dbg-snapshot {
            background:#f7f7f9; border-left:4px solid #c7c7d1;
            padding:10px 12px; margin: 14px 0 10px; font-size:15px;
        }
        .dbg-snap-kv { display:inline-block; margin-right: 14px; }
        .dbg-snap-kv .k { color:#666; }
        .dbg-snap-kv .v { font-weight:700; color:#111; }
        .dbg-json details { margin-top: 10px; }
        .dbg-json summary { cursor: pointer; font-weight: 700; }
        .dbg-json pre {
            background:#111; color:#f2f2f2; padding:12px; border-radius:8px;
            overflow:auto; font-size:13px; line-height:1.45;
        }
        .em { font-style: italic; }

        /* --- Logo header --- */
        .app-logo {
            display: flex;
            justify-content: center;
            margin: 0.5rem 0 1rem;
        }

        /* --- Tabs --- */
        div[data-testid="stTabs"] button {
            padding: 0.25rem 0.5rem;
        }
        @media (max-width: 600px) {
            div[data-testid="stTabs"] > div > div > div[role="tablist"] {
                flex-wrap: wrap;
            }
            div[data-testid="stTabs"] button {
                flex: 1 0 auto;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header():
    st.markdown('<div class="app-logo">', unsafe_allow_html=True)
    st.image("logo.png", width=140)
    st.markdown('</div>', unsafe_allow_html=True)
    st.divider()
