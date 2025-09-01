import pandas as pd
import streamlit as st
from utils.formatting import bold, usd, pct, safe


def build_why_buy_html(row: dict) -> str:
    tkr = safe(row.get("Ticker", ""))
    price = usd(row.get("Price"))
    tp = usd(row.get("TP"))
    res = usd(row.get("Resistance"))
    rr_res = safe(row.get("RR_to_Res", ""))
    rr_tp = safe(row.get("RR_to_TP", ""))
    change_pct = pct(row.get("Change%"))
    relvol = safe(row.get("RelVol(TimeAdj63d)"))
    tp_reward = usd(row.get("TPReward$", None))
    tp_reward_pct = pct(row.get("TPReward%", None))
    daily_atr = usd(row.get("DailyATR", None))
    daily_cap = usd(row.get("DailyCap", None))
    hist_cnt = safe(row.get("Hist21d_PassCount", ""))
    hist_ex = safe(row.get("Hist21d_Examples", ""))
    support_type = safe(row.get("SupportType", ""))
    support_price = usd(row.get("SupportPrice"))
    session = safe(row.get("Session", ""))
    entry_src = safe(row.get("EntrySrc", ""))
    vol_src = safe(row.get("VolSrc", ""))
    header = (
        f"{bold(tkr)} looks attractive here: it last traded near {bold(price)}. "
        f"We’re aiming for a take-profit around {bold(tp)} (halfway to the recent high at {bold(res)}). "
        f"That sets reward-to-risk at roughly {bold(rr_res)}:1 to the recent high and {bold(rr_tp)}:1 to the take-profit."
    )
    bullets = [
        f"- Momentum & liquidity: up {bold(change_pct)} today with relative volume {bold(relvol)} (time-adjusted vs 63-day average).",
        f"- Distance to target: {bold(tp_reward)} ({bold(tp_reward_pct)}). Daily ATR ≈ {bold(daily_atr)}, so a typical month (~21 trading days) allows about {bold(daily_cap)} of movement.",
        f"- History check: {bold(hist_cnt)} instances in the past year where a 21-day move met/exceeded this target. Examples:{hist_ex}.",
        f"- Support: {bold(support_type)} near {bold(support_price)}.",
        f"- Data basis: Session={session} • EntrySrc={entry_src} • VolSrc={vol_src}.",
    ]
    return "<div class='whybuy'>" + header + "<br>" + "<br>".join(bullets) + "</div>"


def outcomes_summary(dfh: pd.DataFrame):
    if dfh is None or dfh.empty:
        st.info("No outcomes yet.")
        return
    n = len(dfh)
    if "result_status" in dfh.columns:
        s_status = dfh["result_status"].astype(str)
    else:
        s_status = pd.Series(["PENDING"] * n, index=dfh.index, dtype="string")
    if "hit" in dfh.columns:
        hit_mask = dfh["hit"].astype(bool)
    else:
        hit_mask = pd.Series([False] * n, index=dfh.index, dtype=bool)
    settled_mask = s_status.eq("SETTLED")
    pending_mask = ~settled_mask
    settled = int(settled_mask.sum())
    pending = int(pending_mask.sum())
    hits = int((settled_mask & hit_mask).sum())
    misses = settled - hits
    st.caption(f"Settled: {settled} • Hits: {hits} • Misses: {misses} • Pending: {pending}")
    sort_cols = [c for c in ["run_date", "ticker"] if c in dfh.columns]
    if sort_cols:
        ascending = [False if c == "run_date" else True for c in sort_cols]
        df_show = dfh.sort_values(sort_cols, ascending=ascending)
    else:
        df_show = dfh
    st.dataframe(df_show, use_container_width=True, height=min(600, 80 + 28 * len(df_show)))
