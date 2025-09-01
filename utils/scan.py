import pandas as pd
import swing_options_screener as sos

def safe_run_scan(with_options: bool = True) -> dict:
    """Call sos.run_scan across historical signature variants and normalize outputs.

    Returns a dict with keys {"pass": DataFrame|None, "scan": DataFrame|None}.
    """
    try:
        out = sos.run_scan(market="sp500", with_options=with_options)
    except TypeError:
        try:
            out = sos.run_scan(universe="sp500", with_options=with_options)
        except TypeError:
            out = sos.run_scan(with_options=with_options)

    df_pass, df_scan = None, None

    if isinstance(out, dict):
        cand = out.get("pass")
        if isinstance(cand, pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df")
        if df_pass is None and isinstance(cand, pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df_unadjusted")
        if df_pass is None and isinstance(cand, pd.DataFrame):
            df_pass = cand

        cand = out.get("scan")
        if isinstance(cand, pd.DataFrame):
            df_scan = cand
        cand = out.get("scan_df")
        if df_scan is None and isinstance(cand, pd.DataFrame):
            df_scan = cand

    elif isinstance(out, (list, tuple)):
        if len(out) >= 1 and isinstance(out[0], pd.DataFrame):
            df_pass = out[0]
        if len(out) >= 2 and isinstance(out[1], pd.DataFrame):
            df_scan = out[1]
    elif isinstance(out, pd.DataFrame):
        df_pass = out

    return {"pass": df_pass, "scan": df_scan}
