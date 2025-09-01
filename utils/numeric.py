"""Numeric utility helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def safe_float(x: Any) -> float:
    """Best effort float conversion handling iterables and NaNs.

    Parameters
    ----------
    x : Any
        Value to convert. If ``x`` is a pandas Series or other iterable,
        the first element is used. ``None`` or invalid inputs result in
        ``numpy.nan``.

    Returns
    -------
    float
        Converted float or ``numpy.nan`` when conversion is not possible.
    """
    if isinstance(x, pd.Series):
        x = x.iloc[0] if not x.empty else np.nan
    elif isinstance(x, (np.ndarray, list, tuple)):
        x = x[0] if len(x) else np.nan

    if pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan

