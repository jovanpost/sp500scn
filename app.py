def _why_buy_card(row: pd.Series) -> str:
    """
    Plain-English rationale. Uses parentheses to introduce technical terms,
    but keeps the narrative understandable for non-traders.
    """
    t = row["Ticker"]
    price = float(row["Price"])
    tp = float(row["TP"])
    res = float(row["Resistance"])
    rr_res = float(row["RR_to_Res"])
    rr_tp = float(row["RR_to_TP"])
    sup_type = str(row["SupportType"])
    sup_px = float(row["SupportPrice"])
    tp_dollar = float(row["TPReward$"])
    tp_pct = float(row["TPReward%"])
    daily_atr = float(row["DailyATR"]) if pd.notna(row.get("DailyATR")) else np.nan
    weekly_atr = daily_atr * 5 if np.isfinite(daily_atr) else np.nan
    monthly_atr = daily_atr * 21 if np.isfinite(daily_atr) else np.nan

    relvol = float(row.get("RelVol(TimeAdj63d)")) if "RelVol(TimeAdj63d)" in row else np.nan
    vol_note = "—"
    if np.isfinite(relvol):
        vol_note = f"about **{(relvol-1)*100:.0f}%** vs typical pace (time-adjusted)."

    # Fix change% double-scaling
    chg = float(row.get("Change%")) if "Change%" in row else np.nan
    chg_display = f"{chg:.2f}%" if np.isfinite(chg) else "—"

    exp = row.get("OptExpiry", "")
    buyk = row.get("BuyK", "")
    sellk = row.get("SellK", "")
    opt_line = ""
    if exp and buyk and sellk:
        opt_line = f" via the **${buyk}/{sellk}** vertical call spread expiring **{exp}**"

    examples = str(row.get("Hist21d_Examples","")).strip()
    pass_count = row.get("Hist21d_PassCount","—")

    md = []
    md.append(
        f"**{t}** is a buy{opt_line} because it recently reached about **${res:.2f} (resistance)** "
        f"and now trades near **${price:.2f} (current price)**. "
        f"That makes the target at **${tp:.2f}** feel realistic."
    )

    md.append("")
    md.append("**Why this setup makes sense**")
    bullets = [
        f"**Reward vs. risk**: from **${sup_px:.2f} (support)** up to resistance is about **{rr_res:.2f}:1** "
        f"(to the nearer target it’s **{rr_tp:.2f}:1**).",
        f"**Move needed to TP:** **${tp_dollar:.2f}** (≈ **{tp_pct:.2f}%**)."
    ]
    if np.isfinite(daily_atr):
        bullets.append(
            f"**Volatility runway (ATR):** Daily ATR is **${daily_atr:.2f}**, "
            f"so over ~21 trading days a typical move could be **~${monthly_atr:.2f}**. "
            f"For context: ~**${weekly_atr:.2f}** in a week."
        )
    if np.isfinite(chg):
        bullets.append(
            f"**Today’s tone & volume:** the stock is up **{chg_display}** today; "
            f"volume is {vol_note}"
        )

    md.extend([f"- {b}" for b in bullets])

    md.append("")
    md.append(
        f"**History check (21 trading days):** need about **{tp_pct:.2f}%** to hit TP. "
        f"Over the past year, **{pass_count}** windows cleared that move."
    )
    if examples:
        md.append("**Examples:**")
        for ex in [e.strip() for e in examples.split(";") if e.strip()]:
            md.append(f"- {ex}")

    ts = row.get("EntryTimeET","")
    if ts:
        md.append("")
        md.append(f"_Data as of **{ts}**._")

    return "\n".join(md)
