# app.py — Swing Options Screener (UNADJUSTED, Finviz-style rel vol)
# - Intraday-aware: uses 1m bars (<=5m fresh) or fast_info; falls back to daily partial during RTH
# - Finviz-style Relative Volume with elapsed minutes scaling (3-mo ≈ 63 sessions)
# - Resistance = prior N-day high (ex-today). TP = halfway to resistance.
# - 2:1 R:R check uses (resistance-entry) / (entry-support).
# - ATR capacity & 21d realism checks.
# - Optional bull call spread suggestion near TP.
# - Streamlit UI: paste tickers or dump text, tweak thresholds, run, download results.

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import re
from datetime import datetime, timedelta, time
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# -------- Timezone (ET) ----------
try:
    from zoneinfo import ZoneInfo  # Py3.9+
except Exception:
    from backports.zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
DELIM = "|"

# -------- Defaults & knobs ----------
SUPPORT_LOOKBACK_DAYS = 21
PIVOT_K = 2
PIVOT_LOOKBACK_DAYS = 40
ATR_STOP_MULT = 1.5
RR_MIN_DEFAULT = 2.0
REL_VOL_MIN_DEFAULT = 1.10
RES_LOOKBACK_DEFAULT = 21
TARGET_OPT_DAYS_DEFAULT = 30
OPT_WIDTH_PREFERENCE = [5.0, 2.5, 1.0, 10.0]

DEFAULT_TICKERS = [
    'ENPH','NCLH','CZR','CCL','DOW','INTC','LUV','SLB','MGM','APA',
    'HST','HAL','SW','KEY','USB','FITB','HPQ','IVZ','HBAN','TFC',
    'WY','LKQ','WBD','AES','RF','DVN','FCX','SMCI','F','BEN',
    'PCG','MRNA','FTV','KIM','BKR','BAX','HPE','OXY','IPG','DOC',
    'CPRT','BF-B','NWSA','BAC','INVH','CNC','KHC','NWS','CAG','IP',
    'CPB','CMG','UDR','CMCSA','CTRA','NI','MTCH','VICI','GEN','HRL',
    'KVUE','EXC','FE','AMCR','PFE','PPL','CNP','MOS','KDP','PSKY',
    'VTRS','KMI','WBA','BMY','VZ','T','CSX'
]

# ===============================
# Utilities
# ===============================
def _to_float(x):
    if isinstance(x, pd.Series):
        x = x.iloc[0] if not x.empty else np.nan
    elif isinstance(x, (np.ndarray, list, tuple)):
        x = x[0] if len(x) else np.nan
    if pd.isna(x): return np.nan
    try: return float(x)
    except Exception: return np.nan

def _fmt_ts(ts: Optional[datetime]) -> str:
    if not isinstance(ts, datetime): return ""
    return ts.strftime("%Y-%m-%d %H:%M:%S ET")

def _now_et() -> datetime:
    return datetime.now(ET)

def _market_session_state() -> Tuple[datetime, datetime, datetime]:
    now = _now_et()
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return now, open_t, close_t

# ===============================
# Data fetch (cached)
# ===============================
@st.cache_data(ttl=300, show_spinner=False)  # 5 min cache
def fetch_daily_history_unadjusted(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.Ticker(ticker).history(period="16mo", auto_adjust=False, actions=False)
        if df is None or df.empty: return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df.columns = [c.title() for c in df.columns]
        for col in ('Open','High','Low','Close','Volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return None

@st.cache_data(ttl=30, show_spinner=False)  # short cache; minute bars change
def fetch_1m_today_unadjusted(ticker: str) -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period="1d", interval="1m", auto_adjust=False, progress=False, prepost=False)
        if df is None or df.empty: return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET).tz_localize(None)
        df.columns = [c.title() for c in df.columns]
        for col in ('Open','High','Low','Close','Volume'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception:
        return None

# ===============================
# Indicators & levels
# ===============================
def _atr_from_ohlc(df: pd.DataFrame, win: int) -> float:
    hl  = df['High'] - df['Low']
    hpc = (df['High'] - df['Close'].shift(1)).abs()
    lpc = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).dropna()
    if len(tr) < win: return np.nan
    return _to_float(tr.rolling(window=win, min_periods=win).mean().iloc[-1])

def _recent_pivot_low(df: pd.DataFrame, k=2, lookback_days=40) -> float:
    if len(df) < 2*k + 1: return np.nan
    need = lookback_days + 2*k + 1
    sub = df.iloc[-need:] if len(df) >= need else df.copy()
    lows = sub['Low'].values
    n = len(sub)
    for idx in range(n - k - 1, k - 1, -1):
        left_ok  = all(lows[idx] < lows[idx - i] for i in range(1, k+1))
        right_ok = all(lows[idx] < lows[idx + i] for i in range(1, k+1) if idx + i < n)
        if left_ok and right_ok:
            return _to_float(lows[idx])
    return np.nan

# ===============================
# Intraday price/volume path
# ===============================
def _intraday_quote_and_volume(ticker: str):
    """Return (last_price, today_cum_volume, source_note, entry_ts_ET). Uses 1m if fresh (<=5m), else fast_info, else NaN."""
    now, open_t, close_t = _market_session_state()
    if not (open_t < now < close_t):
        return np.nan, np.nan, "market_closed", None

    m1 = fetch_1m_today_unadjusted(ticker)
    if m1 is not None and not m1.empty:
        last_ts = m1.index[-1]
        if (now - last_ts).total_seconds() <= 5*60 and np.isfinite(_to_float(m1['Close'].iloc[-1])):
            last_px = _to_float(m1['Close'].iloc[-1])
            today_vol = _to_float(m1['Volume'].sum())
            if np.isfinite(last_px) and today_vol >= 0:
                return last_px, today_vol, "1m_fresh", last_ts

    # fallback: fast_info
    try:
        fi = yf.Ticker(ticker).fast_info
        lp = _to_float(fi.get("last_price", np.nan)) if isinstance(fi, dict) else np.nan
        tv = _to_float(fi.get("last_volume", np.nan)) if isinstance(fi, dict) else np.nan
        if np.isfinite(lp):
            return lp, (tv if np.isfinite(tv) else np.nan), "fast_info", now
    except Exception:
        pass
    return np.nan, np.nan, "no_intraday", None

def _previous_close_robust(df_daily: pd.DataFrame, ticker: str) -> float:
    try:
        fi = yf.Ticker(ticker).fast_info
        pc = _to_float(fi.get('previous_close', np.nan)) if isinstance(fi, dict) else np.nan
        if np.isfinite(pc): return pc
    except Exception:
        pass
    et_today = _now_et().date()
    idx_dates = df_daily.index.date
    if len(df_daily) == 0: return np.nan
    if idx_dates[-1] == et_today and len(df_daily) >= 2:
        return _to_float(df_daily['Close'].iloc[-2])
    return _to_float(df_daily['Close'].iloc[-1])

def get_entry_prevclose_todayvol(df_daily: pd.DataFrame, ticker: str):
    """
    OPEN  : 1m (<=5m) OR fast_info; else daily partial; entry_ts is 'now' in OPEN path.
    CLOSED: official daily close (today if completed) or last close; entry_ts is 16:00 ET of that day.
    """
    now, open_t, close_t = _market_session_state()
    et_today = now.date()
    idx_dates = df_daily.index.date if len(df_daily) else []
    source: Dict[str,str] = {}

    # Market OPEN
    if open_t < now < close_t:
        entry, today_vol, src, entry_ts = _intraday_quote_and_volume(ticker)
        prev_close = _previous_close_robust(df_daily, ticker)

        if np.isfinite(entry) and np.isfinite(prev_close):
            source['session'] = 'OPEN'
            source['entry_src'] = src
            source['vol_src']   = src
            if (src in ('fast_info','no_intraday')) and not np.isfinite(today_vol):
                if len(idx_dates) and idx_dates[-1] == et_today:
                    today_vol = _to_float(df_daily['Volume'].iloc[-1])
                    source['vol_src'] = 'daily_partial'
            if not isinstance(entry_ts, datetime):
                entry_ts = now
            return entry, prev_close, today_vol, source, entry_ts

        # daily partial fallback while OPEN
        if len(idx_dates) and idx_dates[-1] == et_today:
            entry = _to_float(df_daily['Close'].iloc[-1])
            prev_close = _previous_close_robust(df_daily, ticker)
            today_vol = _to_float(df_daily['Volume'].iloc[-1])
            source['session']   = 'OPEN'
            source['entry_src'] = 'daily_partial_close'
            source['vol_src']   = 'daily_partial'
            entry_ts = now
            return entry, prev_close, today_vol, source, entry_ts

        # last resort during OPEN
        if len(df_daily) >= 1:
            entry = _to_float(df_daily['Close'].iloc[-1])
            prev_close = _previous_close_robust(df_daily, ticker)
            today_vol = np.nan
            source['session']   = 'OPEN'
            source['entry_src'] = 'daily_last_close_open_fallback'
            source['vol_src']   = 'unknown'
            entry_ts = now
            return entry, prev_close, today_vol, source, entry_ts

        return np.nan, np.nan, np.nan, {'session':'OPEN','entry_src':'none','vol_src':'none'}, None

    # Market CLOSED
    source['session'] = 'CLOSED'
    if len(df_daily) == 0:
        return np.nan, np.nan, np.nan, source, None

    market_close_time = time(16, 0, 0)
    if len(idx_dates) and idx_dates[-1] == et_today:
        entry = _to_float(df_daily['Close'].iloc[-1])
        prev_close = _to_float(df_daily['Close'].iloc[-2]) if len(df_daily) >= 2 else np.nan
        today_vol = _to_float(df_daily['Volume'].iloc[-1])
        entry_ts = datetime.combine(df_daily.index[-1].date(), market_close_time, tzinfo=ET)
        source['entry_src'] = 'daily_today_close'
        source['vol_src']   = 'daily_today'
        return entry, prev_close, today_vol, source, entry_ts

    entry = _to_float(df_daily['Close'].iloc[-1])
    prev_close = _to_float(df_daily['Close'].iloc[-2]) if len(df_daily) >= 2 else np.nan
    today_vol = _to_float(df_daily['Volume'].iloc[-1])
    last_trading_date = df_daily.index[-1].date()
    entry_ts = datetime.combine(last_trading_date, market_close_time, tzinfo=ET)
    source['entry_src'] = 'daily_last_close'
    source['vol_src']   = 'daily_last'
    return entry, prev_close, today_vol, source, entry_ts

# ===============================
# Finviz-style Relative Volume
# ===============================
def compute_relvol_time_adjusted(df_daily: pd.DataFrame, today_vol: float, use_median=False) -> float:
    """
    RelVol = today_vol / (avg_63 * session_progress)
    avg_63 = average of last 63 completed sessions (≈ 3 months)
    progress = elapsed minutes since open / 390
    """
    if not np.isfinite(today_vol) or today_vol < 0:
        return np.nan

    base_series = df_daily['Volume'].iloc[-64:-1]  # last 63 completed sessions
    if base_series.empty: return np.nan
    avg_63 = _to_float(base_series.median() if use_median else base_series.mean())
    if not np.isfinite(avg_63) or avg_63 <= 0: return np.nan

    now = _now_et()
    open_t  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    if now <= open_t: progress = 0.0
    elif now >= close_t: progress = 1.0
    else:
        progress = (now - open_t).total_seconds() / (close_t - open_t).total_seconds()
        progress = max(progress, 1/390)  # avoid div/0 early

    expected_by_now = avg_63 * progress
    return today_vol / expected_by_now if expected_by_now > 0 else np.nan

# ===============================
# Options helpers (optional)
# ===============================
def _parse_expirations(t: yf.Ticker):
    exps = getattr(t, "options", [])
    out = []
    today_et = _now_et().date()
    for s in exps or []:
        try:
            d = datetime.strptime(s, "%Y-%m-%d").date()
            if d >= today_et:
                out.append(d)
        except Exception:
            continue
    return out

def _nearest_expiration(t: yf.Ticker, target_days=TARGET_OPT_DAYS_DEFAULT):
    exps = _parse_expirations(t)
    if not exps:
        return None, "no_expirations"
    today = _now_et().date()
    best = min(exps, key=lambda d: abs((d - today).days - target_days))
    return best, None

def _mid_or_last(row):
    bid = _to_float(row.get('bid', np.nan))
    ask = _to_float(row.get('ask', np.nan))
    last= _to_float(row.get('lastPrice', np.nan))
    if np.isfinite(bid) and np.isfinite(ask) and ask >= bid:
        return 0.5*(bid+ask)
    if np.isfinite(last):
        return last
    if np.isfinite(bid): return bid
    if np.isfinite(ask): return ask
    return np.nan

def _conservative_price(long_row, short_row):
    ask_long  = _to_float(long_row.get('ask', np.nan))
    bid_short = _to_float(short_row.get('bid', np.nan))
    if not np.isfinite(ask_long):  ask_long  = _mid_or_last(long_row)
    if not np.isfinite(bid_short): bid_short = _mid_or_last(short_row)
    return ask_long - bid_short

def _row_for_strike(df_calls: pd.DataFrame, strike: float):
    exact = df_calls[np.isclose(df_calls['strike'], strike)]
    if not exact.empty:
        return exact.iloc[0]
    idx = (df_calls['strike'] - strike).abs().idxmin()
    return df_calls.loc[idx]

def suggest_bull_call_spread(ticker: str, tp_price: float, target_days=TARGET_OPT_DAYS_DEFAULT, width_pref=OPT_WIDTH_PREFERENCE):
    try:
        t = yf.Ticker(ticker)
        expiry, err = _nearest_expiration(t, target_days=target_days)
        if err: return None, err
        chain = t.option_chain(expiry.strftime("%Y-%m-%d"))
        calls = chain.calls
        if calls is None or calls.empty: return None, "no_calls"

        sell_row = calls.iloc[(calls['strike'] - tp_price).abs().idxmin()]
        sell_k = _to_float(sell_row['strike'])

        buy_row = None; buy_k = None
        for w in width_pref:
            candidate = sell_k - w
            if not calls[np.isclose(calls['strike'], candidate)].empty:
                buy_row = _row_for_strike(calls, candidate)
                buy_k = _to_float(buy_row['strike'])
                break
        if buy_row is None:
            lower_calls = calls[calls['strike'] < sell_k]
            if lower_calls.empty:
                return None, "no_lower_strike"
            buy_row = lower_calls.iloc[(lower_calls['strike'] - (sell_k - 5)).abs().idxmin()]
            buy_k = _to_float(buy_row['strike'])

        mid_long  = _mid_or_last(buy_row)
        mid_short = _mid_or_last(sell_row)
        if not (np.isfinite(mid_long) and np.isfinite(mid_short)):
            return None, "no_quotes"

        debit_mid = mid_long - mid_short
        debit_con = _conservative_price(buy_row, sell_row)
        width = sell_k - buy_k
        max_profit_mid = max(width - debit_mid, 0)
        max_profit_con = max(width - debit_con, 0)
        rr_mid = (max_profit_mid / debit_mid) if debit_mid > 0 else np.nan
        rr_con = (max_profit_con / debit_con) if debit_con > 0 else np.nan
        breakeven_mid = buy_k + debit_mid

        return {
            "OptExpiry": expiry.isoformat(),
            "BuyK": round(buy_k, 2),
            "SellK": round(sell_k, 2),
            "Width": round(width, 2),
            "DebitMid": round(debit_mid, 2),
            "DebitCons": round(debit_con, 2),
            "MaxProfitMid": round(max_profit_mid, 2),
            "MaxProfitCons": round(max_profit_con, 2),
            "RR_Spread_Mid": round(rr_mid, 2) if np.isfinite(rr_mid) else "",
            "RR_Spread_Cons": round(rr_con, 2) if np.isfinite(rr_con) else "",
            "BreakevenMid": round(breakeven_mid, 2),
            "PricingNote": "ask(buy)-bid(sell)=DebitCons; mid-mid=DebitMid"
        }, None
    except Exception as e:
        return None, f"opt_error:{e}"

# ===============================
# Core evaluation
# ===============================
def evaluate_ticker(
    ticker: str,
    res_days=RES_LOOKBACK_DEFAULT,
    rel_vol_min=REL_VOL_MIN_DEFAULT,
    use_relvol_median=False,
    rr_min=RR_MIN_DEFAULT,
    support_lookback_days=SUPPORT_LOOKBACK_DAYS,
    pivot_k=PIVOT_K,
    pivot_lookback_days=PIVOT_LOOKBACK_DAYS,
    prefer_stop="safest",
):
    df = fetch_daily_history_unadjusted(ticker)
    if df is None or df.empty: return None, "no_data"
    if len(df) < max(22, res_days + 1): return None, "insufficient_rows"

    entry, prev_close, today_vol, src, entry_ts = get_entry_prevclose_todayvol(df, ticker)
    if not (np.isfinite(prev_close) and np.isfinite(entry)): return None, "bad_entry_prevclose"
    change = (entry - prev_close) / prev_close
    if change <= 0: return None, "not_up_on_day"

    rel_vol = compute_relvol_time_adjusted(df, today_vol, use_median=use_relvol_median)
    if not (np.isfinite(rel_vol) and rel_vol >= rel_vol_min): return None, "relvol_low_timeadj"

    rolling_high = df['High'].rolling(window=res_days, min_periods=res_days).max()
    resistance = _to_float(rolling_high.shift(1).iloc[-1])  # prior N-day high (exclude today)
    if not (np.isfinite(resistance) and resistance > entry): return None, "no_upside_to_resistance"

    tp = entry + 0.5 * (resistance - entry)
    tp_reward  = tp - entry
    res_reward = resistance - entry

    daily_atr = _atr_from_ohlc(df, 14)
    daily_cap = daily_atr * 21.0 if np.isfinite(daily_atr) else 0.0
    if not (daily_cap > tp_reward): return None, "atr_capacity_short_vs_tp"

    eval_date = df.index[-1]
    past = df[df.index >= (eval_date - timedelta(days=365))].copy()
    if len(past) < 42: return None, "insufficient_past_for_21d"
    fwd_close    = past['Close'].shift(-21)
    fwd_move_pct = (fwd_close - past['Close']) / past['Close'] * 100.0
    tp_req_pct   = (tp_reward / entry) * 100.0
    pass_count   = int((fwd_move_pct >= tp_req_pct).fillna(False).sum())
    if pass_count == 0: return None, "history_21d_zero_pass"

    swing_low_21 = _to_float(df['Low'].iloc[-support_lookback_days:].min())
    pivot_low    = _recent_pivot_low(df, k=pivot_k, lookback_days=pivot_lookback_days)
    atr_stop     = entry - ATR_STOP_MULT * (daily_atr if np.isfinite(daily_atr) else 0.0)

    supports = {
        'SwingLow21': swing_low_21 if np.isfinite(swing_low_21) and swing_low_21 < entry else np.nan,
        'PivotLow':   pivot_low    if np.isfinite(pivot_low)    and pivot_low    < entry else np.nan,
        f'ATR{ATR_STOP_MULT}x': atr_stop if np.isfinite(atr_stop) and atr_stop < entry else np.nan
    }
    valids = [(k, v) for k, v in supports.items() if np.isfinite(v)]
    if not valids: return None, "no_valid_support"

    if prefer_stop == "structure":
        order = ["PivotLow", "SwingLow21", f"ATR{ATR_STOP_MULT}x"]
        chosen = None
        for k in order:
            if np.isfinite(supports.get(k, np.nan)):
                chosen = (k, supports[k]); break
        if chosen is None: chosen = max(valids, key=lambda kv: kv[1])
    else:
        chosen = max(valids, key=lambda kv: kv[1])
    support_type, stop = chosen

    risk = entry - stop
    if risk <= 0: return None, "non_positive_risk"

    rr_to_res = res_reward / risk
    if rr_to_res < rr_min: return None, "rr_to_res_below_min"
    rr_to_tp = tp_reward / risk

    # Historical examples (top 3)
    top = fwd_move_pct[fwd_move_pct >= tp_req_pct].dropna().sort_values(ascending=False).head(3)
    examples_str = "; ".join([f"{pd.to_datetime(dt).date()}:+{pct:.2f}%" for dt, pct in top.items()])

    row = {
        'Ticker': ticker,
        'EvalDate': df.index[-1].date().isoformat(),
        'Price': round(entry, 2),
        'EntryTimeET': _fmt_ts(entry_ts) if entry_ts else "",
        'Change%': round(change * 100.0, 2),
        'RelVol(TimeAdj63d)': round(rel_vol, 2),
        'Resistance': round(resistance, 2),
        'TP': round(tp, 2),
        'RR_to_Res': round(rr_to_res, 2),
        'RR_to_TP': round(rr_to_tp, 2),
        'SupportType': support_type,
        'SupportPrice': round(stop, 2),
        'Risk$': round(risk, 4),
        'TPReward$': round(tp_reward, 4),
        'TPReward%': round((tp_reward / entry) * 100.0, 2),
        'DailyATR': round(daily_atr, 4) if np.isfinite(daily_atr) else "",
        'DailyCap': round(daily_cap, 4),
        'Hist21d_PassCount': pass_count,
        'Hist21d_Max%': round(_to_float(top.max()), 2) if not top.empty else "",
        'Hist21d_Examples': examples_str,
        # diagnostics
        'Session': src.get('session',''),
        'EntrySrc': src.get('entry_src',''),
        'VolSrc': src.get('vol_src',''),
    }
    return row, None

# ===============================
# Parsing helpers (UI)
# ===============================
def parse_ticker_text(text: str) -> List[str]:
    if not text: return []
    raw = [t.strip().upper() for t in text.replace("\n", ",").replace(" ", ",").split(",") if t.strip()]
    out, seen = [], set()
    for t in raw:
        if re.fullmatch(r"[A-Z][A-Z0-9\-.]{0,4}", t) and t not in seen:
            seen.add(t); out.append(t)
    return out

def extract_tickers_from_blob(blob: str) -> List[str]:
    if not blob: return []
    cands = re.findall(r"\b[A-Z]{1,5}\b(?:\.[A-Z]{1,2})?", blob)
    # Filter obvious non-tickers (very rough)
    bad = {"USA","Page","Total","Filters","ET","EPS","PE","P/E","INC","BIL","B"}
    tickers = [t for t in cands if t not in bad]
    # uniq preserve order
    seen=set(); out=[]
    for t in tickers:
        if t not in seen:
            seen.add(t); out.append(t)
    return out

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Swing Options Screener", layout="wide")
st.title("📊 Swing Options Screener (Finviz-style RelVol, Unadjusted)")

with st.sidebar:
    st.subheader("Settings")
    res_days = st.number_input("Resistance lookback (days)", min_value=10, max_value=60, value=RES_LOOKBACK_DEFAULT, step=1)
    relvol_min = st.slider("Min Relative Volume (time-adjusted)", 1.0, 2.0, REL_VOL_MIN_DEFAULT, 0.05)
    use_median = st.checkbox("RelVol: use median (robust to outliers)", value=False)
    rr_min = st.slider("Min R:R to Resistance", 1.0, 4.0, RR_MIN_DEFAULT, 0.1)
    stop_mode = st.selectbox("Stop preference", ["safest","structure"], index=0, help="safest = highest valid support; structure = PivotLow > SwingLow > ATR")
    include_options = st.checkbox("Suggest bull call spread near TP", value=False)
    opt_days = st.number_input("Target option days", min_value=7, max_value=90, value=TARGET_OPT_DAYS_DEFAULT, step=1)

    st.caption("Prices & volumes are typically **~15 min delayed** on Yahoo.")

col1, col2 = st.columns(2)
with col1:
    tickers_text = st.text_area("Tickers (comma/space/newline separated):", ", ".join(DEFAULT_TICKERS[:30]), height=140)
with col2:
    with st.expander("Or paste a Finviz table / blob to extract tickers"):
        blob = st.text_area("Paste here:")
        if st.button("Extract tickers from blob"):
            extracted = extract_tickers_from_blob(blob)
            st.success(f"Found {len(extracted)} tickers.")
            if extracted:
                st.code(", ".join(extracted))

run_btn = st.button("▶️ Run Screener", type="primary")

if run_btn:
    user_tickers = parse_ticker_text(tickers_text)
    if not user_tickers and blob:
        user_tickers = extract_tickers_from_blob(blob)

    if not user_tickers:
        st.warning("No tickers provided.")
    else:
        rows = []
        fails = []
        prog = st.progress(0.0, text="Scanning…")
        for i, t in enumerate(user_tickers):
            row, reason = evaluate_ticker(
                t,
                res_days=res_days,
                rel_vol_min=relvol_min,
                use_relvol_median=use_median,
                rr_min=rr_min,
                support_lookback_days=SUPPORT_LOOKBACK_DAYS,
                pivot_k=PIVOT_K,
                pivot_lookback_days=PIVOT_LOOKBACK_DAYS,
                prefer_stop=stop_mode,
            )
            if reason is None:
                if include_options:
                    opt, err = suggest_bull_call_spread(t, row['TP'], target_days=opt_days)
                    if err is None and opt:
                        row.update(opt)
                    else:
                        row.update({
                            "OptExpiry":"", "BuyK":"", "SellK":"", "Width":"",
                            "DebitMid":"", "DebitCons":"", "MaxProfitMid":"", "MaxProfitCons":"",
                            "RR_Spread_Mid":"", "RR_Spread_Cons":"", "BreakevenMid":"", "PricingNote": err or ""
                        })
                rows.append(row)
            else:
                fails.append({"Ticker": t, "Reason": reason})
            prog.progress((i+1)/len(user_tickers))

        st.write(f"Processed at {_now_et().strftime('%Y-%m-%d %H:%M:%S ET')}")

        if rows:
            df = pd.DataFrame(rows).sort_values(["Price","Ticker"])
            st.success(f"{len(df)} PASS tickers")
            st.dataframe(df, use_container_width=True)

            # download buttons
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="pass_tickers.csv", mime="text/csv")
        else:
            st.warning("No PASS tickers found.")

        if fails:
            with st.expander("Show FAIL reasons"):
                st.dataframe(pd.DataFrame(fails), use_container_width=True)

# Footer note
st.caption("© Your Screener — data via Yahoo Finance (delayed). This is not financial advice.")
