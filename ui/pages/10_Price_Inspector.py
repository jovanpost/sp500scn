from __future__ import annotations

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Price Inspector (minimal)", page_icon="🔍", layout="wide")

from data_lake.storage import Storage, load_prices_cached, validate_prices_schema

st.title("🔍 Price Inspector (minimal)")

col0, col1, col2 = st.columns([1, 1, 1])
with col0:
    tkr = st.text_input("Ticker", value="NVDA").strip().upper()
with col1:
    day = st.date_input("Date", value=pd.Timestamp("2020-03-20").date())
with col2:
    pad = st.number_input("± days window", min_value=0, max_value=10, value=2, step=1)

if st.button("Load"):
    storage = Storage.from_env()
    parquet_path = f"prices/{tkr}.parquet"
    parquet_exists = storage.exists(parquet_path)
    st.caption(f"Parquet present? **{parquet_exists}**  ·  Path: `{parquet_path}`")
    if not parquet_exists:
        st.warning("File not found; try another ticker or check ingest.")

    start = pd.Timestamp(day) - pd.Timedelta(days=int(pad))
    end = pd.Timestamp(day) + pd.Timedelta(days=int(pad))

    df = load_prices_cached(
        storage,
        cache_salt=storage.cache_salt(),
        tickers=[tkr],
        start=start,
        end=end,
    )

    if df.empty:
        st.error("No rows loaded from the lake in this window.")
    else:
        # Guardrails on read
        try:
            validate_prices_schema(df)
        except Exception as e:
            st.error(f"Validator FAILED: {e}")

        st.write(
            f"Rows: {len(df)}   ·   Range: {str(df['date'].min())} → {str(df['date'].max())}"
        )
        st.caption(f"Rows for {tkr}: {len(df[df['Ticker'] == tkr])}")
        st.dataframe(df, use_container_width=True)

        exact = df[df["date"] == pd.Timestamp(day)]
        st.subheader("Exact day (lake)")
        if exact.empty:
            st.warning("No exact row for that date (see table above).")
        else:
            st.dataframe(exact, use_container_width=True)
            r = exact.iloc[0]
            diag = {
                "Ticker": r["Ticker"],
                "date": str(pd.to_datetime(r["date"]).date()),
                "Open": float(r["Open"]),
                "High": float(r["High"]),
                "Low": float(r["Low"]),
                "Close": float(r["Close"]),
                "Adj Close": float(r["Adj Close"]) if pd.notna(r["Adj Close"]) else None,
                "Dividends": float(r.get("Dividends", 0) or 0),
                "Stock Splits": float(r.get("Stock Splits", 0) or 0),
                "Volume": int(r["Volume"]) if pd.notna(r["Volume"]) else None,
            }
            st.json(diag)
