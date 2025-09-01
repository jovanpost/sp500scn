# SP500SCN

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

