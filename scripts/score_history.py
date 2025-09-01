#!/usr/bin/env python3
# scripts/score_history.py
# - Reads data/history/outcomes.csv
# - For Outcome == PENDING, checks if TargetLevel was hit (High >= level)
#   between EvalDate (inclusive) and min(WindowEnd, today) (inclusive).
#   Results are written back to outcomes.csv.
from _bootstrap import add_repo_root; add_repo_root()

from utils.io import OUTCOMES_CSV
from utils.outcomes import evaluate_outcomes, read_outcomes, write_outcomes


def main() -> None:
    df = read_outcomes(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to score.")
        return

    new_df = evaluate_outcomes(df, mode="historical")
    write_outcomes(new_df, OUTCOMES_CSV)
    print(f"Scored {len(new_df)} rows → wrote outcomes.csv")


if __name__ == "__main__":
    main()

