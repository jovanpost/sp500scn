# app_history_reader.py
import pandas as pd

def _raw_base(owner: str, repo: str, branch: str = "data") -> str:
    return f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}"

def fetch_history_index(owner: str, repo: str, branch: str = "data") -> pd.DataFrame:
    url = _raw_base(owner, repo, branch) + "/history/index.csv"
    return pd.read_csv(url)

def fetch_run_csv(owner: str, repo: str, csv_rel_path: str, branch: str = "data") -> pd.DataFrame:
    base = _raw_base(owner, repo, branch)
    url = f"{base}/{csv_rel_path}"
    return pd.read_csv(url)

def fetch_latest(owner: str, repo: str, branch: str = "data") -> pd.DataFrame:
    idx = fetch_history_index(owner, repo, branch)
    if idx.empty:
        return idx
    # pick most recent run
    idx["__order"] = pd.to_datetime(idx["RunTimeET"], errors="coerce")
    idx = idx.sort_values("__order")
    latest_path = idx.iloc[-1]["CSVPath"]
    return fetch_run_csv(owner, repo, latest_path, branch)
