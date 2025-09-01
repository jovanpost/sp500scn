import pandas as pd
import swing_options_screener as sos


def safe_run_scan(with_options: bool = True) -> dict:
    """Run the screener across historical signatures and normalize outputs.

    Parameters
    ----------
    with_options: bool, default True
        Whether to ask the engine for options fields. Falls back gracefully
        if the underlying ``run_scan`` does not support this argument.

    Returns
    -------
    dict
        Mapping with keys ``pass`` and ``scan`` holding DataFrames or ``None``.
    """
    import pandas as _pd

    try:
        out = sos.run_scan(market="sp500", with_options=with_options)
    except TypeError:
        try:
            out = sos.run_scan(universe="sp500", with_options=with_options)
        except TypeError:
            try:
                out = sos.run_scan(with_options=with_options)
            except TypeError:
                out = sos.run_scan()

    df_pass, df_scan = None, None

    if isinstance(out, dict):
        cand = out.get("pass")
        if isinstance(cand, _pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df")
        if df_pass is None and isinstance(cand, _pd.DataFrame):
            df_pass = cand
        cand = out.get("pass_df_unadjusted")
        if df_pass is None and isinstance(cand, _pd.DataFrame):
            df_pass = cand

        cand = out.get("scan")
        if isinstance(cand, _pd.DataFrame):
            df_scan = cand
        cand = out.get("scan_df")
        if df_scan is None and isinstance(cand, _pd.DataFrame):
            df_scan = cand

    elif isinstance(out, (list, tuple)):
        if len(out) >= 1 and isinstance(out[0], _pd.DataFrame):
            df_pass = out[0]
        if len(out) >= 2 and isinstance(out[1], _pd.DataFrame):
            df_scan = out[1]

    elif isinstance(out, _pd.DataFrame):
        df_pass = out

    return {"pass": df_pass, "scan": df_scan}
