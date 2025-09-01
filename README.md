# SP500SCN

## Hit Checking

Use the `scripts/check_hits.py` utility to update `data/history/outcomes.csv` with trade results.

```bash
python scripts/check_hits.py
```

The script fetches daily price data for each pending entry and marks hits or misses accordingly.

## Historical Scoring

For backtesting datasets that include `EvalDate`, `WindowEnd`, and `TargetLevel`
columns, run `scripts/score_history.py` to evaluate whether price targets were
hit before each window closed.

```bash
python scripts/score_history.py
```
