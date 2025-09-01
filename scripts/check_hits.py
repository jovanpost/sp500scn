#!/usr/bin/env python3
"""Update ``data/history/outcomes.csv`` by evaluating pending rows."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import OUTCOMES_CSV
from utils.outcomes import evaluate_outcomes, read_outcomes, write_outcomes


def main() -> None:
    if not OUTCOMES_CSV.exists():
        print("No outcomes.csv yet; nothing to check.")
        return

    df = read_outcomes(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to check.")
        return

    df = evaluate_outcomes(df, mode="pending")
    write_outcomes(df, OUTCOMES_CSV)
    print("Updated outcomes.csv")


if __name__ == "__main__":
    main()

