#!/usr/bin/env python3
"""Evaluate data/history/outcomes.csv in either pending or historical mode."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io import OUTCOMES_CSV
from utils.outcomes import evaluate_outcomes, read_outcomes, write_outcomes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate outcomes.csv")
    p.add_argument(
        "--mode",
        choices=("pending", "historical"),
        default="pending",
        help="Evaluation mode: 'pending' checks open trades, 'historical' scores backtests.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not OUTCOMES_CSV.exists():
        print("No outcomes.csv yet; nothing to evaluate.")
        return

    df = read_outcomes(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to evaluate.")
        return

    new_df = evaluate_outcomes(df, mode=args.mode)
    write_outcomes(new_df, OUTCOMES_CSV)
    print(f"Evaluated outcomes.csv in {args.mode} mode")


if __name__ == "__main__":
    main()
