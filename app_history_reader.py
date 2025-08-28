# app_history_reader.py
# Helper for the Streamlit app to load historical runs written by the Action.

import glob
import os
import pandas as pd

# Prefer the new GitHub Actions output location; fall back to legacy if present.
HISTORY_GLOBS = [
    "data/history/pass_*.psv",   # new path (written by schedule.yml)
    "history/pass_*.psv",        # legacy fallback
]

def load_history_df() -> pd.DataFrame:
    """Return a DataFrame concatenating all historical PASS snapshots."""
    paths = []
    for pat in HISTORY_GLOBS:
        paths.extend(sorted(glob.glob(pat)))

    if not paths:
        return pd.DataFrame()

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p, sep="|")
            df["__source_file"] = os.path.basename(p)
            frames.append(df)
        except Exception:
            # Skip unreadable files; keep going
            continue

    if not frames:
        return pd.DataFrame()

    # Sort newest first by file name if timestamp is embedded in filename;
    # otherwise the app can sort by EvalDate.
    out = pd.concat(frames, ignore_index=True)
    return out
