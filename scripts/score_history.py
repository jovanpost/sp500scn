#!/usr/bin/env python3
# scripts/score_history.py
# - Reads data/history/outcomes.csv
# - For Outcome == PENDING, checks if TargetLevel was hit (High >= level)
#   between EvalDate (inclusive) and min(WindowEnd, today) (inclusive).
#   Results are written back to outcomes.csv.
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

HIST_DIR = os.path.join(ROOT, "data", "history")
OUTCOMES_CSV = os.path.join(HIST_DIR, "outcomes.csv")

from utils.outcomes import read_outcomes, score_history, write_outcomes


def main() -> None:
    if not os.path.exists(OUTCOMES_CSV):
        print("No outcomes.csv yet; nothing to score.")
        return

    df = read_outcomes(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to score.")
        return

    new_df = score_history(df)
    write_outcomes(new_df, OUTCOMES_CSV)
    print(f"Scored {len(new_df)} rows → wrote outcomes.csv")


if __name__ == "__main__":
    main()

