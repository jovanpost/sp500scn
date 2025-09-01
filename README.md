# SP500SCN

## Hit Checking

Use the `scripts/check_hits.py` utility to update `data/history/outcomes.csv` with trade results.

```bash
python scripts/check_hits.py
```

The script fetches daily price data for each pending entry and marks hits or misses accordingly.
