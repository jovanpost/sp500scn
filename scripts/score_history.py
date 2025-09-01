#!/usr/bin/env python3
# scripts/score_history.py
# - Reads data/history/outcomes.csv
# - For Outcome == PENDING, checks if TargetLevel was hit (High >= level)
#   between EvalDate (inclusive) and min(WindowEnd, today) (inclusive).
#   Results are written back to outcomes.csv.
from utils.io import OUTCOMES_PATH, read_csv, write_csv
from utils.outcomes import score_history


def main() -> None:
    if not OUTCOMES_PATH.exists():
        print("No outcomes.csv yet; nothing to score.")
        return

    df = read_csv(OUTCOMES_PATH)
    if df.empty:
        print("outcomes.csv empty; nothing to score.")
        return

    new_df = score_history(df)
    write_csv(OUTCOMES_PATH, new_df)
    print(f"Scored {len(new_df)} rows → wrote outcomes.csv")


if __name__ == "__main__":
    main()

