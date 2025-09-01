#!/usr/bin/env python3
"""Update ``data/history/outcomes.csv`` by evaluating pending rows."""

from utils.io import OUTCOMES_PATH, read_csv, write_csv
from utils.outcomes import check_pending_hits


def main() -> None:
    if not OUTCOMES_PATH.exists():
        print("No outcomes.csv yet; nothing to check.")
        return

    df = read_csv(OUTCOMES_PATH)
    if df.empty:
        print("outcomes.csv empty; nothing to check.")
        return

    df = check_pending_hits(df)
    write_csv(OUTCOMES_PATH, df)
    print("Updated outcomes.csv")


if __name__ == "__main__":
    main()

