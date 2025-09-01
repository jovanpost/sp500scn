def _bold(x):
    return f"<b>{x}</b>"


def _usd(x, nd=2):
    try:
        return f"${x:.{nd}f}"
    except Exception:
        return str(x)


def _pct(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return str(x)


def _safe(x):
    return str(x) if x is not None else ""
