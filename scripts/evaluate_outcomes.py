#!/usr/bin/env python3
"""Evaluate and update ``data/history/outcomes.csv`` in pending or historical mode."""

from _bootstrap import add_repo_root; add_repo_root()

import argparse

from utils.io import OUTCOMES_CSV
from utils.outcomes import evaluate_outcomes, read_outcomes, write_outcomes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate outcomes.csv in pending or historical mode",
    )
    p.add_argument(
        "--mode",
        choices=["pending", "historical"],
        default="pending",
        help="Evaluation mode: 'pending' updates open positions; 'historical' scores past datasets.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = read_outcomes(OUTCOMES_CSV)
    if df.empty:
        print("outcomes.csv empty; nothing to evaluate.")
        return

    df = evaluate_outcomes(df, mode=args.mode)
    write_outcomes(df, OUTCOMES_CSV)
    print(f"Updated outcomes.csv (mode={args.mode})")


if __name__ == "__main__":
    main()
