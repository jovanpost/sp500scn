import math


def bold(x):
    return f"<b>{x}</b>"


def usd(x, nd=2):
    try:
        return f"${float(x):.{nd}f}"
    except Exception:
        return str(x)


def pct(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)


def safe(x):
    return str(x) if x is not None else ""


def fmt_ts_et(ts):
    try:
        return ts.strftime("%Y-%m-%d %H:%M:%S %Z") if hasattr(ts, "strftime") else str(ts)
    except Exception:
        return str(ts)
