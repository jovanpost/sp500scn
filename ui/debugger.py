import json
import html as _html
import pandas as pd
import streamlit as st
import swing_options_screener as sos
from utils.tickers import normalize_symbol
from utils.formatting import _usd, _pct


# Functions extracted from app.py for diagnostics

def _fmt_ts_et(ts):
    try:
        return ts.strftime("%Y-%m-%d %H:%M:%S %Z") if hasattr(ts, "strftime") else str(ts)
    except Exception:
        return str(ts)


def _num(x, nd=4):
    try:
        return float(x)
    except Exception:
        return None


def _finite(x):
    from math import isfinite
    try:
        return isfinite(float(x))
    except Exception:
        return False


def _mk_reason_expl(reason: str, ctx: dict) -> str:
    """Turn engine reason codes into friendly, numeric explanations."""
    lines = []
    code = (reason or "").strip()

    chg = ctx.get("change_pct")
    rel = ctx.get("relvol")
    rel_min = ctx.get("relvol_min")
    entry = ctx.get("entry")
    prev_close = ctx.get("prev_close")
    res = ctx.get("resistance")
    tp = ctx.get("tp")
    daily_atr = ctx.get("daily_atr")
    daily_cap = ctx.get("daily_cap")
    req_tp_pct = ctx.get("tp_req_pct")

    if code == "relvol_low_timeadj":
        lines.append(
            f"Relative volume is too low: current RelVol (time-adjusted) is **{rel:.2f}×**, "
            f"but the minimum is **{rel_min:.2f}×**."
        )
    elif code == "not_up_on_day":
        lines.append(
            f"Price isn’t up on the day: change is **{_pct(chg)}** from "
            f"yesterday’s close {_usd(prev_close)} to entry {_usd(entry)}."
        )
    elif code == "no_upside_to_resistance":
        lines.append(f"No room to the recent high: resistance {_usd(res)} is not above entry {_usd(entry)}.")
    elif code == "atr_capacity_short_vs_tp":
        lines.append(
            "ATR capacity is too small to reasonably reach the target in a month: "
            f"need ≈ **{_pct(req_tp_pct)}** to target (≈ {_usd(tp)}), but Daily ATR is "
            f"{_usd(daily_atr, nd=4)}, implying about **{_usd(daily_cap)}** over ~21 trading days."
        )
    elif code == "history_21d_zero_pass":
        lines.append(
            "History check failed: in the last year there were **0** cases where a 21-trading-day move "
            f"matched or exceeded the required **{_pct(req_tp_pct)}**."
        )
    elif code in {"no_valid_support", "non_positive_risk"}:
        lines.append("Couldn’t find a valid support below price to place a stop (risk would be non-positive).")
    elif code == "rr_to_res_below_min":
        lines.append("Reward-to-risk to the recent high is below the minimum (needs ≥ 2:1).")
    elif code in {"insufficient_rows", "insufficient_past_for_21d"}:
        lines.append("Not enough price history to evaluate this ticker robustly.")
    elif code == "bad_entry_prevclose":
        lines.append("Intraday quote/previous close unavailable or inconsistent.")
    elif code == "no_data":
        lines.append("No price data returned for this input after normalization.")
    else:
        lines.append(f"Engine rejected the setup: **{code}**.")

    snap = []
    if _finite(entry) and _finite(prev_close):
        snap.append(f"Entry {_usd(entry)} vs prev close {_usd(prev_close)} → day change {_pct(chg)}.")
    if _finite(res) and _finite(tp):
        snap.append(f"Resistance {_usd(res)}, TP {_usd(tp)}.")
    if _finite(rel):
        snap.append(f"RelVol (time-adjusted): {rel:.2f}× (min {rel_min:.2f}×).")
    if _finite(daily_atr):
        snap.append(f"Daily ATR {_usd(daily_atr, nd=4)} ⇒ ~{_usd(daily_cap)} / 21 trading days.")
    if snap:
        lines.append("")
        lines.append("**Snapshot:** " + " ".join(snap))
    return "<br>".join(lines)


def diagnose_ticker(
    ticker: str,
    res_days=None,
    rel_vol_min=None,
    relvol_median=False,
    rr_min=None,
    stop_mode="safest",
):
    """Return title, details dict, and explanation for UI."""
    res_days = res_days if res_days is not None else getattr(sos, "RES_LOOKBACK_DEFAULT", 21)
    rel_vol_min = rel_vol_min if rel_vol_min is not None else getattr(sos, "REL_VOL_MIN_DEFAULT", 1.50)
    rr_min = rr_min if rr_min is not None else getattr(sos, "RR_MIN_DEFAULT", 2.0)

    original = (ticker or "").strip()
    symbol = normalize_symbol(original)

    df = sos._get_history(symbol) if symbol else None
    if df is not None and df.empty:
        st.warning(
            f"No normalized OHLCV found for {symbol}. Check that prices/{symbol}.parquet exists and has date/open/high/low/close/volume."
        )
        df = None

    entry = prev_close = today_vol = None
    src = {}
    entry_ts = None

    if df is not None:
        try:
            entry, prev_close, today_vol, src, entry_ts = sos.get_entry_prevclose_todayvol(df, symbol)
        except Exception:
            try:
                entry = float(df["Close"].iloc[-1])
                prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else None
                today_vol = float(df["Volume"].iloc[-1])
                src = {"session": "UNKNOWN", "entry_src": "fallback", "vol_src": "fallback"}
                entry_ts = df.index[-1].to_pydatetime() if hasattr(df.index, "to_pydatetime") else None
            except Exception:
                df = None

    if df is None:
        title = f"{original if original else '—'} FAILED ❌ — no_data"
        narrative = (
            "No price data returned for this input. Try the **ticker symbol** or a known S&P-500 "
            "company name (the Debugger maps names like ‘NVIDIA’ → ‘NVDA’, ‘Tesla’ → ‘TSLA’)."
        )
        details = {
            "entry": None,
            "prev_close": None,
            "today_vol": None,
            "src": {},
            "entry_ts": _fmt_ts_et(entry_ts),
            "relvol_time_adj": None,
            "resistance": None,
            "tp": None,
            "daily_atr": None,
            "daily_cap": None,
            "explanation_md": narrative,
        }
        return title, details

    row, reason = sos.evaluate_ticker(
        symbol,
        res_days=res_days,
        rel_vol_min=rel_vol_min,
        use_relvol_median=relvol_median,
        rr_min=rr_min,
        prefer_stop=stop_mode,
    )

    if reason is None and isinstance(row, dict):
        narrative = (
            f"**{symbol} PASSED** ✔️ — price is up **{row.get('Change%', 0):.2f}%** today, "
            f"time-adjusted RelVol **{row.get('RelVol(TimeAdj63d)', 0):.2f}×**. "
            f"Target {row.get('TP')} vs price {row.get('Price')}, "
            f"Daily ATR ≈ {row.get('DailyATR')} (~{row.get('DailyCap')} per month). "
            f"R:R to high ≈ **{row.get('RR_to_Res')}**:1; to TP ≈ **{row.get('RR_to_TP')}**:1."
        )
        details = {
            "entry": entry,
            "prev_close": prev_close,
            "today_vol": today_vol,
            "src": src,
            "entry_ts": _fmt_ts_et(entry_ts),
            "explanation_md": narrative,
        }
        return f"{symbol} PASSED ✅", details

    from math import isfinite

    ctx = {}
    ctx["entry"] = _num(entry)
    ctx["prev_close"] = _num(prev_close)
    ctx["change_pct"] = (
        (ctx["entry"] - ctx["prev_close"]) / ctx["prev_close"] * 100.0
        if (_finite(ctx["entry"]) and _finite(ctx["prev_close"]) and ctx["prev_close"] != 0)
        else None
    )

    relvol_val = None
    try:
        if df is not None and _finite(today_vol):
            relvol_val = sos.compute_relvol_time_adjusted(df, today_vol, use_median=relvol_median)
    except Exception:
        relvol_val = None
    ctx["relvol"] = relvol_val
    ctx["relvol_min"] = rel_vol_min

    try:
        if df is not None and len(df) >= max(22, res_days + 1) and _finite(ctx["entry"]):
            rolling_high = df["High"].rolling(window=res_days, min_periods=res_days).max()
            res = float(rolling_high.shift(1).iloc[-1])
            ctx["resistance"] = res
            if isfinite(res) and res > ctx["entry"]:
                ctx["tp"] = ctx["entry"] + 0.5 * (res - ctx["entry"])
                if isfinite(ctx["tp"]):
                    ctx["tp_req_pct"] = (ctx["tp"] - ctx["entry"]) / ctx["entry"] * 100.0
    except Exception:
        pass

    try:
        if df is not None:
            da = sos._atr_from_ohlc(df, 14)
            ctx["daily_atr"] = da
            if _finite(da):
                ctx["daily_cap"] = da * 21.0
    except Exception:
        pass

    title = f"{symbol} FAILED ❌ — {reason}"
    narrative = _mk_reason_expl(reason, ctx)
    details = {
        "entry": entry,
        "prev_close": prev_close,
        "today_vol": today_vol,
        "src": src,
        "entry_ts": _fmt_ts_et(entry_ts),
        "relvol_time_adj": float(relvol_val) if _finite(relvol_val) else None,
        "resistance": ctx.get("resistance"),
        "tp": ctx.get("tp"),
        "daily_atr": ctx.get("daily_atr"),
        "daily_cap": ctx.get("daily_cap"),
        "explanation_md": narrative,
    }
    return title, details


def render_debugger_tab():
    st.markdown(
        """
        <style>
          .dbg-wrap { margin-top: 0.5rem; }
          .dbg-title { font-weight: 700; font-size: 1.05rem; margin-bottom: .25rem; }
          .dbg-badge { padding: 2px 6px; border-radius: 6px; font-size: .8rem; margin-left: .4rem; }
          .dbg-badge.pass { background: #0f5132; color: #fff; }
          .dbg-badge.fail { background: #842029; color: #fff; }
          .dbg-subtle { margin-bottom: .5rem; line-height: 1.4; }
          .dbg-snapshot { background: #111; color: #eee; padding: 10px; border-radius: 8px; font-size: .9rem; }
          .dbg-snap-kv { display: inline-block; margin-right: 14px; margin-top: 4px; }
          .dbg-snap-kv .k { color: #bbb; }
          .dbg-snap-kv .v { color: #fff; font-weight: 600; }
          .dbg-json details { margin-top: .5rem; }
          .dbg-json pre { background: #0b0b0b; color: #e6e6e6; padding: 10px; border-radius: 8px; overflow-x: auto; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.header("Debugger")
    dbg_ticker = st.text_input("Enter ticker to debug", key="dbg_ticker_input")

    if dbg_ticker:
        title, details = diagnose_ticker(dbg_ticker.strip().upper())
        is_fail = "FAIL" in (title or "").upper()
        badge = (
            '<span class="dbg-badge fail">FAIL</span>'
            if is_fail
            else '<span class="dbg-badge pass">PASS</span>'
        )

        def g(d, k, default="—"):
            try:
                v = d.get(k, default)
                return default if v is None else v
            except Exception:
                return default

        entry = g(details, "entry")
        prev_close = g(details, "prev_close")
        today_vol = g(details, "today_vol")
        src = g(details, "src", {})
        session = g(src, "session", "—") if isinstance(src, dict) else "—"
        entry_src = g(src, "entry_src", "—") if isinstance(src, dict) else "—"
        vol_src = g(src, "vol_src", "—") if isinstance(src, dict) else "—"
        entry_ts = g(details, "entry_ts")
        resistance = g(details, "resistance")
        tp = g(details, "tp")
        relvol_timeadj = g(details, "relvol_time_adj")
        daily_atr = g(details, "daily_atr")
        daily_cap = g(details, "daily_cap")

        narrative_html = details.get("explanation_md", "")

        html_top = f"""
        <div class="dbg-wrap">
          <div class="dbg-title">{title} {badge}</div>
          <div class="dbg-subtle">{narrative_html}</div>
        """

        html_snapshot = f"""
          <div class="dbg-snapshot">
            <span class="dbg-snap-kv"><span class="k">Session:</span> <span class="v">{session}</span></span>
            <span class="dbg-snap-kv"><span class="k">Entry src:</span> <span class="v">{entry_src}</span></span>
            <span class="dbg-snap-kv"><span class="k">Vol src:</span> <span class="v">{vol_src}</span></span><br/>
            <span class="dbg-snap-kv"><span class="k">Entry:</span> <span class="v">{entry}</span></span>
            <span class="dbg-snap-kv"><span class="k">Prev Close:</span> <span class="v">{prev_close}</span></span>
            <span class="dbg-snap-kv"><span class="k">Today Vol:</span> <span class="v">{today_vol}</span></span><br/>
            <span class="dbg-snap-kv"><span class="k">Resistance:</span> <span class="v">{resistance}</span></span>
            <span class="dbg-snap-kv"><span class="k">TP:</span> <span class="v">{tp}</span></span>
            <span class="dbg-snap-kv"><span class="k">RelVol(Adj):</span> <span class="v">{relvol_timeadj}</span></span>
            <span class="dbg-snap-kv"><span class="k">Daily ATR:</span> <span class="v">{daily_atr}</span></span>
            <span class="dbg-snap-kv"><span class="k">Daily Cap:</span> <span class="v">{daily_cap}</span></span><br/>
            <span class="dbg-snap-kv"><span class="k">Timestamp:</span> <span class="v">{entry_ts}</span></span>
          </div>
        """

        pretty = json.dumps(
            {k: v for k, v in details.items() if k != "explanation_md"},
            indent=2,
            default=str,
        )
        html_json = f"""
          <div class="dbg-json">
            <details>
              <summary>Show raw JSON</summary>
              <pre>{_html.escape(pretty)}</pre>
            </details>
          </div>
        </div>
        """

        st.markdown(html_top, unsafe_allow_html=True)
        st.markdown(html_snapshot, unsafe_allow_html=True)
        st.markdown(html_json, unsafe_allow_html=True)
