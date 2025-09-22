# SP500SCN

## Data Source

Price history is loaded from Supabase Storage parquet files under
`lake/prices/{TICKER}.parquet`. Legacy database table reads and `yfinance`
imports have been removed in favor of this storage-based approach.

Storage paths are normalised automatically: the configured bucket name or
leading slashes are stripped from the object key before Supabase requests are
issued. Environment variables such as `LAKE_PRICES_PREFIX` may therefore be
specified as `prices`, `/prices`, or even `lake/prices` without breaking remote
fetches. The loader logs the resolved bucket, prefix, and key whenever a request
fails to assist with troubleshooting.

## Stock Scanner (Shares Only)

The Streamlit app now includes a **Stock Scanner (Shares Only)** tab that
simulates whole-share entries capped at **$1,000** per trade. Candidates must
pass configurable filters for yesterday's performance, open gap, volume
multiple, and support/resistance ratio. Two exit models are available:

- **ATR multiples** – take-profit/stop-loss are set from Wilder ATR values
  (defaults 1×/1×).
- **Support/Resistance** – exits at the detected support/resistance levels.

Each position is evaluated forward for up to 30 trading days. If neither the
take-profit nor stop-loss triggers, the trade exits at the day-30 close with a
`timeout` label. The tab reports per-trade details (entry/exit, shares, TP/SL,
cost, proceeds, P&L) and aggregates total capital deployed, total P&L, and win
rate. Results can be downloaded as a CSV ledger for further analysis.

## Outcome Evaluation

Use the `scripts/evaluate_outcomes.py` utility to update `data/history/outcomes.csv` with trade results.

`outcomes.csv` tracks columns such as `Ticker`, `EvalDate`, `Price`, `LastPrice`,
`LastPriceAt`, `PctToTarget`, option strikes, status fields, and other notes.

For pending trades:

```bash
python scripts/evaluate_outcomes.py --mode pending
```

For historical datasets that include `EvalDate`, `WindowEnd`, and `TargetLevel` columns:

```bash
python scripts/evaluate_outcomes.py --mode historical
```

The script fetches daily price data for each entry and marks hits or misses accordingly.

