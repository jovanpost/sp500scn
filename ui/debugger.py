import swing_options_screener as sos
from utils.formatting import usd, pct, fmt_ts_et
from utils.data import yf_fetch_daily

_ALIASES = {
    "NVIDIA": "NVDA", "NVIDIA CORPORATION": "NVDA",
    "TESLA": "TSLA", "TESLA INC": "TSLA",
    "APPLE": "AAPL", "APPLE INC": "AAPL",
    "MICROSOFT": "MSFT", "MICROSOFT CORPORATION": "MSFT",
    "ALPHABET": "GOOGL", "GOOGLE": "GOOGL",
    "META": "META", "META PLATFORMS": "META",
    "AMAZON": "AMZN", "AMAZONCOM": "AMZN", "AMAZON.COM": "AMZN",
    "NETFLIX": "NFLX", "WALMART": "WMT", "WALMART INC": "WMT",
    "JPMORGAN": "JPM", "JPMORGAN CHASE": "JPM",
    "BERKSHIRE": "BRK.B", "BERKSHIRE HATHAWAY": "BRK.B",
    "UNITEDHEALTH": "UNH", "UNITEDHEALTH GROUP": "UNH",
    "COCA COLA": "KO", "COCA-COLA": "KO",
    "PEPSICO": "PEP", "ADOBE": "ADBE", "INTEL": "INTC",
    "AMD": "AMD", "BROADCOM": "AVGO", "SALESFORCE": "CRM",
    "SERVICENOW": "NOW", "SERVICE NOW": "NOW",
    "CROWDSTRIKE": "CRWD", "MCDONALDS": "MCD", "MCDONALD'S": "MCD",
    "COSTCO": "COST", "HOME DEPOT": "HD",
    "PROCTER & GAMBLE": "PG", "PROCTER AND GAMBLE": "PG",
    "ELI LILLY": "LLY", "ABBVIE": "ABBV",
    "EXXON": "XOM", "EXXONMOBIL": "XOM", "CHEVRON": "CVX",
}


def _normalize_brk(s: str) -> str | None:
    s2 = s.replace(" ", "").replace("-", "").replace("_", "").upper()
    if s2 in {"BRKB", "BRK.B"}:
        return "BRK.B"
    if s2 in {"BRKA", "BRK.A"}:
        return "BRK.A"
    sU = s.upper().strip()
    if sU in {"BRK B", "BRK-B", "BRK_B"}:
        return "BRK.B"
    if sU in {"BRK A", "BRK-A", "BRK_A"}:
        return "BRK.A"
    return None


def _normalize_symbol(inp: str) -> str | None:
    if not inp:
        return None
    s = str(inp).strip()
    if not s:
        return None
    if 1 <= len(s) <= 6 and all(c.isalnum() or c == "." for c in s):
        brk = _normalize_brk(s)
        return brk if brk else s.upper()
    key = s.upper()
    key = key.replace(",", "").replace(".", "")
    for kill in (" INC", " CORPORATION", " COMPANY", " HOLDINGS", " PLC", " LTD"):
        key = key.replace(kill, "")
    key = key.replace(" CLASS A", "").replace(" CLASS B", "")
    key = " ".join(key.split())
    if key in _ALIASES:
        return _ALIASES[key]
    brk = _normalize_brk(s)
    return brk if brk else s.upper()


def _mk_reason_expl(reason: str, ctx: dict) -> str:
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
            f"Price isn’t up on the day: change is **{pct(chg)}** from "
            f"yesterday’s close {usd(prev_close)} to entry {usd(entry)}."
        )
    elif code == "no_upside_to_resistance":
        lines.append(f"No room to the recent high: resistance {usd(res)} is not above entry {usd(entry)}.")
    elif code == "atr_capacity_short_vs_tp":
        lines.append(
            "ATR capacity is too small to reasonably reach the target in a month: "
            f"need ≈ **{pct(req_tp_pct)}** to target (≈ {usd(tp)}), but Daily ATR is "
            f"{usd(daily_atr, nd=4)}, implying about **{usd(daily_cap)}** over ~21 trading days."
        )
    elif code == "history_21d_zero_pass":
        lines.append(
            "History check failed: in the last year there were **0** cases where a 21-trading-day move "
            f"matched or exceeded the required **{pct(req_tp_pct)}**."
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
    from math import isfinite
    entry_f = ctx.get("entry")
    prev_close_f = ctx.get("prev_close")
    if entry_f is not None and prev_close_f is not None and isfinite(entry_f) and isfinite(prev_close_f):
        snap.append(f"Entry {usd(entry)} vs prev close {usd(prev_close)} → day change {pct(chg)}.")
    if isfinite(ctx.get("resistance")) and isfinite(ctx.get("tp")):
        snap.append(f"Resistance {usd(res)}, TP {usd(tp)}.")
    if isfinite(rel):
        snap.append(f"RelVol (time-adjusted): {rel:.2f}× (min {rel_min:.2f}×).")
    if isfinite(daily_atr):
        snap.append(f"Daily ATR {usd(daily_atr, nd=4)} ⇒ ~{usd(daily_cap)} / 21 trading days.")
    if snap:
        lines.append("")
        lines.append("**Snapshot:** " + " ".join(snap))
    return "<br>".join(lines)


def diagnose_ticker(ticker: str, res_days=None, rel_vol_min=None, relvol_median=False, rr_min=None, stop_mode="safest"):
    res_days = res_days if res_days is not None else getattr(sos, "RES_LOOKBACK_DEFAULT", 21)
    rel_vol_min = rel_vol_min if rel_vol_min is not None else getattr(sos, "REL_VOL_MIN_DEFAULT", 1.10)
    rr_min = rr_min if rr_min is not None else getattr(sos, "RR_MIN_DEFAULT", 2.0)
    original = (ticker or "").strip()
    symbol = _normalize_symbol(original)
    df = sos._get_history(symbol) if symbol else None
    if df is None and symbol:
        df = yf_fetch_daily(symbol)
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
            "entry_ts": fmt_ts_et(entry_ts),
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
            "entry_ts": fmt_ts_et(entry_ts),
            "explanation_md": narrative,
        }
        return f"{symbol} PASSED ✅", details
    from math import isfinite
    ctx = {}
    ctx["entry"] = float(entry) if entry is not None else None
    ctx["prev_close"] = float(prev_close) if prev_close is not None else None
    ctx["change_pct"] = (
        (ctx["entry"] - ctx["prev_close"]) / ctx["prev_close"] * 100.0
        if (ctx["entry"] is not None and ctx["prev_close"] and ctx["prev_close"] != 0)
        else None
    )
    relvol_val = None
    try:
        if df is not None and today_vol is not None:
            relvol_val = sos.compute_relvol_time_adjusted(df, today_vol, use_median=relvol_median)
    except Exception:
        relvol_val = None
    ctx["relvol"] = relvol_val
    ctx["relvol_min"] = rel_vol_min
    try:
        if df is not None and len(df) >= max(22, res_days + 1) and ctx["entry"] is not None:
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
            if isfinite(da):
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
        "entry_ts": fmt_ts_et(entry_ts),
        "relvol_time_adj": relvol_val,
        "resistance": ctx.get("resistance"),
        "tp": ctx.get("tp"),
        "daily_atr": ctx.get("daily_atr"),
        "daily_cap": ctx.get("daily_cap"),
        "explanation_md": narrative,
    }
    return title, details
