# SP500SCN

## Hit Checking

Use the `scripts/check_hits.py` utility to update `data/history/outcomes.csv` with trade results.

```bash
python scripts/check_hits.py
```

The script fetches daily price data for each pending entry and marks hits or misses accordingly.

## Historical PASS snapshots

Use the `load_history_df` helper in `ui.history` to load historical PASS run
snapshots for the Streamlit app. It concatenates all `pass_*.psv` files from
`data/history/` (falling back to legacy `history/`) into a single DataFrame.
