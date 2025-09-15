    return f"mode={self.mode} bucket={self.bucket}"

    def cache_salt(self) -> str:
        return f"mode={self.mode}|url={self.supabase_url or ''}"

    def selftest(self) -> dict[str, Any]:
        return {"ok": True, "mode": self.mode}

    def list_all(self, prefix: str) -> List[str]:
        """
        Return list of file paths directly under `prefix` for both local and Supabase.

        Notes:
        - Some Supabase SDK builds don't accept keyword `prefix=`, so we prefer positional.
        - Some return an object with `.data`; others return a plain list — we handle both.
        """
        norm = prefix.rstrip("/")

        if self.mode == "supabase":
            items: Iterable | None = None
            api = None
            if self.supabase_client is not None:
                api = self.supabase_client.storage.from_(self.bucket)
            elif hasattr(self.bucket, "list"):
                # legacy fallback
                api = self.bucket

            if api is not None:
                try:
                    # Prefer positional (compatible across versions)
                    res = api.list(norm)
                except TypeError:
                    # Fallback to kwarg if positional signature fails
                    res = api.list(prefix=norm)

                items = getattr(res, "data", res)

            out: List[str] = []
            for it in items or []:
                if isinstance(it, dict):
                    name = it.get("name")
                else:
                    name = getattr(it, "name", None) or str(it)
                if name:
                    out.append(f"{norm}/{name}")
            return sorted(out)

        # local
        base = self._norm(norm)
        if not base.exists():
            return []
        return [f"{norm}/{p.name}" for p in sorted(base.iterdir()) if p.is_file()]

    @classmethod
    def from_env(cls) -> "Storage":
        return cls()


# ====== Cached Parquet loader (conflict-free & backward compatible) ======

@st.cache_data(hash_funcs={Storage: lambda _: 0}, show_spinner=False)
def load_prices_cached(
    _storage: "Storage",
    cache_salt: str | list[str] = "",
    tickers: list[str] | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV for `tickers` from object storage *Parquet* files only (no SQL).

    Accepts BOTH calling styles:
      1) load_prices_cached(storage, tickers, start, end)
      2) load_prices_cached(storage, cache_salt="...", tickers=[...], start=..., end=...)

    Output columns: ['date','Open','High','Low','Close','Adj Close','Volume','Ticker']
    Dates are tz-naive pandas Timestamps. Rows are de-duped (last write wins).
    """
    # --- Backward-compat for positional calls: (storage, tickers, start, end) ---
    # If caller used the old signature, `cache_salt` will actually be a list of tickers.
    if tickers is None and isinstance(cache_salt, (list, tuple)):
        tickers = list(cache_salt)
        cache_salt = ""

    tickers = list(tickers or [])
    if not tickers:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for t in tickers:
        path = f"prices/{t}.parquet"

        # Best-effort existence check (works with both local & supabase)
        exists_fn = getattr(_storage, "exists", None)
        has_file = bool(exists_fn(path)) if callable(exists_fn) else True
        if not has_file:
            continue

        # Load parquet via helper if present, else bytes->parquet
        try:
            raw = _storage.read_parquet_df(path)  # repo helper
        except Exception:
            raw = pd.read_parquet(io.BytesIO(_storage.read_bytes(path)))

        tidy = _tidy_prices(raw, ticker=t)  # must exist in this module

        # Date window filter on index (tidy returns DatetimeIndex)
        if start is not None:
            s = pd.Timestamp(start)
            try:
                s = s.tz_localize(None)
            except Exception:
                pass
            tidy = tidy[tidy.index >= s]
        if end is not None:
            e = pd.Timestamp(end)
            try:
                e = e.tz_localize(None)
            except Exception:
                pass
            tidy = tidy[tidy.index <= e]

        # Keep 'date' as a column for downstream code
        tidy = tidy.reset_index().rename(columns={"index": "date"})
        tidy["date"] = pd.to_datetime(tidy["date"], errors="coerce")
        try:
            tidy["date"] = tidy["date"].dt.tz_localize(None)
        except Exception:
            pass
        tidy = tidy.dropna(subset=["date"])

        frames.append(tidy)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Ensure expected columns exist and order them
    cols = ["date", "Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    out = out[cols]

    # Deduplicate per (date, Ticker): keep last
    out = (
        out.sort_values(["Ticker", "date"])
           .drop_duplicates(["Ticker", "date"], keep="last")
           .reset_index(drop=True)
    )

    return out 