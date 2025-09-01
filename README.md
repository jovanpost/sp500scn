# SP500SCN

## Hit Checking

Use the `scripts/check_hits.py` utility to update `data/history/outcomes.csv` with trade results.

```bash
python scripts/check_hits.py
```

The script fetches daily price data for each pending entry and marks hits or misses accordingly.

## Historical Scoring

To evaluate whether historical targets were met within their windows, use:

```bash
python scripts/score_history.py
```

Rows are marked `YES` or `NO` depending on whether the target level was reached before the window expired.
