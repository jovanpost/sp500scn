# SP500SCN

## Outcome Evaluation

Use `scripts/evaluate_outcomes.py` to update `data/history/outcomes.csv`.
The script reads the existing outcomes file, evaluates rows and writes the
results back to disk.

```bash
# Check pending trades for TP hits or expiry
python scripts/evaluate_outcomes.py --mode pending

# Score historical backtest rows
python scripts/evaluate_outcomes.py --mode historical
```
