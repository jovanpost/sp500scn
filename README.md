# SP500SCN

## Hit Checking

Trade outcomes are evaluated by `utils.outcomes.evaluate_outcomes`, which can
process both pending trades and historical window checks. The repository
provides small command line helpers that read `data/history/outcomes.csv`, apply
`evaluate_outcomes`, and write the results back:

```bash
python scripts/check_hits.py      # evaluate pending trades
python scripts/score_history.py   # evaluate historical windows
```

Both scripts fetch daily price data and mark hits or misses as appropriate.
