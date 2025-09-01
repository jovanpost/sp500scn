#!/usr/bin/env python3
"""Update ``data/history/outcomes.csv`` by evaluating pending rows."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
HIST_DIR = os.path.join(REPO_ROOT, "data", "history")
OUT_PATH = os.path.join(HIST_DIR, "outcomes.csv")

from utils.outcomes import check_pending_hits, read_outcomes, write_outcomes


def main() -> None:
    if not os.path.exists(OUT_PATH):
        print("No outcomes.csv yet; nothing to check.")
        return

    df = read_outcomes(OUT_PATH)
    if df.empty:
        print("outcomes.csv empty; nothing to check.")
        return

    df = check_pending_hits(df)
    write_outcomes(df, OUT_PATH)
    print("Updated outcomes.csv")


if __name__ == "__main__":
    main()

