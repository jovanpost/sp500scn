#!/usr/bin/env python3
"""Update ``data/history/outcomes.csv`` by evaluating pending rows."""

from _bootstrap import add_repo_root; add_repo_root()

from utils.io import OUTCOMES_CSV
from utils.outcomes import check_pending_hits, read_outcomes, write_outcomes


def main() -> None:
    if not OUTCOMES_CSV.exists():
        print("No outcomes.csv yet; nothing to check.")
        return

    df = read_outcomes(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to check.")
        return

    df = check_pending_hits(df)
    write_outcomes(df, OUTCOMES_CSV)
    print("Updated outcomes.csv")


if __name__ == "__main__":
    main()

