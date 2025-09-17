# RAW OHLC policy

* Store raw provider OHLCV values as traded; keep the vendor-supplied `Adj Close` column alongside them.
* Never rescale or back-adjust `Open`, `High`, `Low`, or `Close` after ingest.
* Stable schema for parquet files: `date`, `Ticker`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`, `Dividends`, `Stock Splits`.
* The validator enforces that when corporate actions occur, `Close` must differ from `Adj Close`.
