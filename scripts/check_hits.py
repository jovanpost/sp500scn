#!/usr/bin/env python3
"""Update ``data/history/outcomes.csv`` by evaluating pending rows."""

from _bootstrap import add_repo_root; add_repo_root()

from utils.io import OUTCOMES_CSV
from utils.outcomes import evaluate_outcomes, read_outcomes, write_outcomes


def main() -> None:
    df = read_outcomes(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to check.")
        return

    df = evaluate_outcomes(df, mode="pending")
    write_outcomes(df, OUTCOMES_CSV)
    print("Updated outcomes.csv")


if __name__ == "__main__":
    main()

