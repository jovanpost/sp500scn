import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler
from utils.formatting import _bold, _usd, _pct, _safe
from utils.scan import safe_run_scan
from .history import _apply_dark_theme


def _style_negatives(df: pd.DataFrame) -> Styler:
    """Return a Styler adding class "neg" to negative numeric cells."""
    classes = pd.DataFrame("", index=df.index, columns=df.columns)
    num_cols = df.select_dtypes(include="number").columns
    for col in num_cols:
        classes.loc[df[col] < 0, col] = "neg"
    return df.style.set_td_classes(classes)


def build_why_buy_html(row: dict) -> str:
    tkr = _safe(row.get("Ticker", ""))
    price = _usd(row.get("Price"))
    tp = _usd(row.get("TP"))
    res = _usd(row.get("Resistance"))
    rr_res = _safe(row.get("RR_to_Res", ""))
    rr_tp = _safe(row.get("RR_to_TP", ""))
    change_pct = _pct(row.get("Change%"))
    relvol = _safe(row.get("RelVol(TimeAdj63d)"))
    tp_reward = _usd(row.get("TPReward$", None))
    tp_reward_pct = _pct(row.get("TPReward%", None))
    daily_atr = _usd(row.get("DailyATR", None))
    daily_cap = _usd(row.get("DailyCap", None))
    hist_cnt = _safe(row.get("Hist21d_PassCount", ""))
    hist_ex = _safe(row.get("Hist21d_Examples", ""))
    support_type = _safe(row.get("SupportType", ""))
    support_price = _usd(row.get("SupportPrice"))
    session = _safe(row.get("Session", ""))
    entry_src = _safe(row.get("EntrySrc", ""))
    vol_src = _safe(row.get("VolSrc", ""))

    header = (
        f"{_bold(tkr)} looks attractive here: it last traded near {_bold(price)}. "
        f"We’re aiming for a take-profit around {_bold(tp)} (halfway to the recent high at {_bold(res)}). "
        f"That sets reward-to-risk at roughly {_bold(rr_res)}:1 to the recent high and {_bold(rr_tp)}:1 to the take-profit."
    )

    bullets = [
        f"- Momentum & liquidity: up {_bold(change_pct)} today with relative volume {_bold(relvol)} (time-adjusted vs 63-day average).",
        f"- Distance to target: {_bold(tp_reward)} ({_bold(tp_reward_pct)}). Daily ATR ≈ {_bold(daily_atr)}, "
        f"so a typical month (~21 trading days) allows about {_bold(daily_cap)} of movement.",
        f"- History check: {_bold(hist_cnt)} instances in the past year where a 21-day move met/exceeded this target. Examples: {hist_ex}.",
        f"- Support: {_bold(support_type)} near {_bold(support_price)}.",
        f"- Data basis: Session={session} • EntrySrc={entry_src} • VolSrc={vol_src}.",
    ]

    return "<div class='whybuy'>" + header + "<br>" + "<br>".join(bullets) + "</div>"


def _sheet_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a sheet-friendly subset (subset of columns)."""
    prefer = [
        "Ticker","EvalDate","Price","EntryTimeET",
        "Change%","RelVol(TimeAdj63d)","Resistance","TP",
        "RR_to_Res","RR_to_TP","SupportType","SupportPrice",
        "Risk$","TPReward$","TPReward%","ResReward$","ResReward%",
        "DailyATR","DailyCap","Hist21d_PassCount"
    ]
    cols = [c for c in prefer if c in df.columns]
    return df.loc[:, cols].copy() if cols else df.copy()


def _render_why_buy_block(df: pd.DataFrame):
    """Render WHY BUY expanders per ticker."""
    if df is None or df.empty:
        return
    st.markdown("### WHY BUY details")
    for _, row in df.iterrows():
        tkr = str(row.get("Ticker", "")).strip() or "—"
        with st.expander(f"WHY BUY — {tkr}", expanded=False):
            html = build_why_buy_html(row)
            st.markdown(html, unsafe_allow_html=True)


def render_scanner_tab():
    st.markdown("#### Scanner")

    # Red RUN button (custom CSS inside this tab to avoid global collisions)
    st.markdown(
        """
        <style>
        div.stButton > button:first-child {
            background-color: red !important;
            color: white !important;
            font-weight: bold !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    run_clicked = st.button("RUN", key="run_scan_btn")

    if run_clicked:
        with st.spinner("Scanning…"):
            out = safe_run_scan()
        df_pass: pd.DataFrame | None = out.get("pass", None)

        st.session_state["last_pass"] = df_pass

        if df_pass is None or df_pass.empty:
            st.warning("No tickers passed the filters.")
        else:
            st.success(f"Found {len(df_pass)} passing tickers (latest run).")
            st.markdown(
                _apply_dark_theme(_style_negatives(df_pass)).to_html(),
                unsafe_allow_html=True,
            )
            _render_why_buy_block(df_pass)
            with st.expander("Google-Sheet style view (optional)", expanded=False):
                st.markdown(
                    _apply_dark_theme(
                        _style_negatives(_sheet_friendly(df_pass))
                    ).to_html(),
                    unsafe_allow_html=True,
                )

    elif isinstance(st.session_state.get("last_pass"), pd.DataFrame) and not st.session_state["last_pass"].empty:
        df_pass: pd.DataFrame = st.session_state["last_pass"]
        st.info(f"Showing last run in this session • {len(df_pass)} tickers")
        st.markdown(
            _apply_dark_theme(_style_negatives(df_pass)).to_html(),
            unsafe_allow_html=True,
        )
        _render_why_buy_block(df_pass)
        with st.expander("Google-Sheet style view (optional)", expanded=False):
            st.markdown(
                _apply_dark_theme(
                    _style_negatives(_sheet_friendly(df_pass))
                ).to_html(),
                unsafe_allow_html=True,
            )
    else:
        st.caption("No results yet. Press **RUN** to scan.")
