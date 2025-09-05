# Data Lake Layout (Phase 1)

This repository bootstraps a tiny data lake containing point-in-time
membership information for the S&P 500 and daily adjusted OHLCV prices
for all historical members.

## Storage paths

Files are written either to a Supabase Storage bucket (when the relevant
credentials are available via `st.secrets`) or to a local `.lake/`
folder.

```
membership/sp500_members.parquet
prices/{TICKER}.parquet
manifest/manifest.json
```

The repository also stores a small preview CSV at
`data_lake/sp500_members_preview.csv` for transparency.

## Schemas

### Membership

```
ticker        string
name          string
start_date    date  (inclusive)
end_date      date  or null (exclusive)
source        string
notes         string or null
```

### Prices

```
date, open, high, low, close, adj_close, volume, ticker
```

## Rebuild steps

1. Configure Supabase (optional) by creating `.streamlit/secrets.toml`
   from `.streamlit/secrets.toml.example`.
2. Run the app and open the **Data Lake (Phase 1)** tab.
3. Click **Build membership** to fetch and store the historical
   membership table.
4. Click **Ingest prices (batch)** to download daily prices. Use the
   form controls to limit the date range or number of tickers.

The manifest at `manifest/manifest.json` summarises each ingestion run.
Future phases will enable point-in-time joins between membership and
prices.
