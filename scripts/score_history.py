#!/usr/bin/env python3
# scripts/score_history.py
# - Reads data/history/outcomes.csv
# - For Outcome == PENDING, checks if TargetLevel was hit (High >= level)
#   between EvalDate (inclusive) and min(WindowEnd, today) (inclusive).
#   Results are written back to outcomes.csv.
from _bootstrap import add_repo_root; add_repo_root()

from utils.io import OUTCOMES_CSV
from utils.outcomes import read_outcomes, score_history, write_outcomes


def main() -> None:
    if not OUTCOMES_CSV.exists():
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

