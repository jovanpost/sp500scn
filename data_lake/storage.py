def _tidy_prices(df: pd.DataFrame, ticker: str | None = None) -> pd.DataFrame:
    """Normalize OHLCV schema to Yahoo-style columns and drop duplicates.

    Output columns (when available):
      date, Open, High, Low, Close, Adj Close, Volume, Ticker
    """
    df = df.copy()

    # If both variants exist, drop the lowercase one to avoid a clash after rename
    if "ticker" in df.columns and "Ticker" in df.columns:
        df = df.drop(columns=["ticker"])

    # Standardize column names
    rename_map = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adjclose": "Adj Close",
        "adj_close": "Adj Close",
        "adj close": "Adj Close",
        "ticker": "Ticker",
        "Date": "date",
        "Datetime": "date",
        "timestamp": "date",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Ensure we have a 'date' column (pull from index if needed)
    if "date" not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={"index": "date"})
        elif "time" in df.columns:
            df = df.rename(columns={"time": "date"})

    # Coerce to datetime and strip timezone to keep joins simple
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        # Remove timezone if present
        if getattr(df["date"].dtype, "tz", None) is not None:
            df["date"] = df["date"].dt.tz_localize(None)

    # Ensure required OHLCV columns exist
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col not in df.columns:
            df[col] = pd.NA

    # Ensure Adj Close exists
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df.get("Close")

    # Normalize Ticker
    if "Ticker" in df.columns:
        df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    elif ticker is not None:
        df["Ticker"] = str(ticker).upper()

    # Drop rows without a valid date
    if "date" in df.columns:
        df = df.dropna(subset=["date"])

    # Drop duplicate rows on (date, Ticker); keep the last (often adjusted values)
    if "date" in df.columns and "Ticker" in df.columns:
        df = df.sort_values("date").drop_duplicates(subset=["date", "Ticker"], keep="last")

    # Order columns (keep only those that exist)
    cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    df = df[[c for c in cols if c in df.columns]]

    return df