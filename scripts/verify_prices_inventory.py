#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable, Tuple
from pathlib import Path

import requests

# Optional supabase client; used only to sanity-check connectivity or signed URLs.
try:
    from supabase import create_client  # type: ignore
except Exception:
    create_client = None  # type: ignore

# ---- basic ticker utilities --------------------------------------------------

def load_tickers(src: str | None) -> list[str]:
    if not src:
        # Minimal built-in fallback; replace with your S&P source if needed.
        return []
    p = Path(src)
    lines = [ln.strip() for ln in p.read_text().splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln or ln.startswith("#"):
            continue
        out.append(ln.split(",")[0].strip().upper())
    return out

def canonical(t: str) -> str:
    return t.strip().upper().replace(".", "_").replace("-", "_").replace(" ", "_")

# ---- prefix/layout helpers (match app logic) --------------------------------

def resolve_prices_prefix(bucket: str, env_prefix: str | None) -> str:
    raw = (env_prefix or "lake/prices").strip().strip("/")
    b = (bucket or "lake").strip().strip("/")
    if b and raw == b:
        return ""
    if b and raw.startswith(f"{b}/"):
        raw = raw[len(b) + 1 :]
    return raw.strip("/")

def public_object_url(supabase_url: str, bucket: str, path: str) -> str:
    base = supabase_url.rstrip("/")
    b = bucket.strip("/")
    p = path.lstrip("/")
    return f"{base}/storage/v1/object/public/{b}/{p}"


def _auth_session(api_key: str | None) -> requests.Session:
    session = requests.Session()
    if api_key:
        session.headers.update(
            {
                "apikey": api_key,
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
    return session


def _build_prices_path(prefix: str, ticker: str) -> str:
    p = prefix.strip("/")
    if p.endswith("prices"):
        base = p
    elif p:
        base = f"{p}/prices"
    else:
        base = "prices"
    return f"{base}/{ticker}.parquet"

# ---- list & head checks ------------------------------------------------------

def list_prices(bucket_url: str, bucket: str, prefix: str, api_key: str | None) -> set[str]:
    """
    Return set of full paths (e.g. 'prices/WMT.parquet') listed under `prefix`.
    Uses the official storage HTTP endpoint with POST + JSON body.
    """
    pfx = prefix.strip("/")
    url = f"{bucket_url.rstrip('/')}/storage/v1/object/list/{bucket}"
    limit = 1000
    offset = 0
    seen: set[str] = set()
    session = _auth_session(api_key)

    while True:
        body = {
            "prefix": f"{pfx}/" if pfx else "",
            "limit": limit,
            "offset": offset,
            "sortBy": {"column": "name", "order": "asc"},
        }
        r = session.post(url, json=body, timeout=30)
        if r.status_code in (401, 403):
            raise RuntimeError(f"Storage list denied ({r.status_code}). Provide SUPABASE_KEY.")
        r.raise_for_status()
        items = r.json() or []
        new = 0
        for it in items:
            name = it.get("name")
            if not name:
                continue
            full = f"{pfx}/{name}" if pfx else name
            if not full.endswith("/") and full not in seen:
                seen.add(full)
                new += 1
        if len(items) < limit or new == 0:
            break
        offset += limit

    return seen

def head_exists(url: str) -> Tuple[bool, str]:
    """
    HEAD the public object URL. Returns (exists, status_code_or_err).
    """
    try:
        r = requests.head(url, allow_redirects=True, timeout=10)
        if r.status_code == 200:
            return True, "200"
        # Some CDNs return 302 to the object; follow once with GET Range: bytes=0-0
        if r.status_code in (301, 302, 307, 308):
            r2 = requests.get(url, headers={"Range": "bytes=0-0"}, timeout=15)
            return (r2.status_code in (200, 206)), str(r2.status_code)
        return False, str(r.status_code)
    except Exception as e:
        return False, f"ERR:{type(e).__name__}"

# ---- main verification -------------------------------------------------------

def verify(supabase_url: str, bucket: str, tickers: Iterable[str], prefix_env: str | None, out_csv: str):
    prefix = resolve_prices_prefix(bucket, prefix_env)
    bucket_url = supabase_url.rstrip("/")
    supabase_key = os.getenv("SUPABASE_KEY", "").strip() or None

    listing = list_prices(bucket_url, bucket, prefix, supabase_key)

    results = []
    list_present = 0
    head_present = 0
    list_false_neg = 0
    not_found = 0

    for raw in tickers:
        t = raw.strip().upper()
        if not t:
            continue
        path = _build_prices_path(prefix, t)

        listed = (path in listing)
        if listed:
            list_present += 1

        url = public_object_url(bucket_url, bucket, path)
        exists, status = head_exists(url)
        if exists:
            head_present += 1

        status_str = (
            "PRESENT_BOTH" if listed and exists else
            "PRESENT_BY_LIST" if listed and not exists else
            "PRESENT_BY_HEAD" if exists and not listed else
            "NOT_FOUND"
        )
        if status_str == "PRESENT_BY_HEAD":
            list_false_neg += 1
        if status_str == "NOT_FOUND":
            not_found += 1

        results.append({
            "ticker": t,
            "path": path,
            "listed": int(listed),
            "head": int(exists),
            "status": status_str,
            "detail": status,
        })

    both = sum(1 for r in results if r["listed"] and r["head"])

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ticker", "path", "listed", "head", "status", "detail"])
        w.writeheader()
        w.writerows(results)

    total = len(results)
    print(f"== Inventory summary ==")
    print(f"Total tickers:          {total}")
    print(f"Listed present:         {list_present}")
    print(f"HEAD present:           {head_present}")
    print(f"Present in both:        {both}")
    print(f"List false negatives:   {list_false_neg}")
    print(f"Not found:              {not_found}")
    print(f"CSV written to:         {out_csv}")
    # Print a quick sample of mismatches:
    mism = [r for r in results if r["status"] in ("PRESENT_BY_HEAD", "PRESENT_BY_LIST")]
    if mism:
        print("\nExamples of mismatches:")
        for r in mism[:10]:
            print(f" - {r['ticker']}: {r['status']} :: {r['path']} ({r['detail']})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers-file", type=str, help="Path to newline-separated tickers (e.g., S&P 505).")
    ap.add_argument("--out", type=str, default="verify_prices_inventory.csv")
    args = ap.parse_args()

    SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
    SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "lake").strip() or "lake"
    LAKE_PRICES_PREFIX = os.getenv("LAKE_PRICES_PREFIX", "lake/prices")

    if not SUPABASE_URL:
        print("ERROR: SUPABASE_URL not set", file=sys.stderr)
        sys.exit(2)

    tickers = load_tickers(args.tickers_file)
    if not tickers:
        print("WARNING: no tickers provided; checking only the 3 ad-hoc symbols WMT, ADSK, BIIB.")
        tickers = ["WMT", "ADSK", "BIIB"]

    verify(SUPABASE_URL, SUPABASE_BUCKET, tickers, LAKE_PRICES_PREFIX, args.out)

if __name__ == "__main__":
    main()
