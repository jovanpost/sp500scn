import sys
from pathlib import Path
from contextlib import contextmanager

import pandas as pd
from pandas.io.formats.style import Styler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.history as history
import ui.scan as scan


def test_outcomes_summary_orders_columns(monkeypatch):
    df = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "EvalDate": ["2024-01-01"],
            "Price": [10],
            "RelVol(TimeAdj63d)": [1.2],
            "LastPrice": [11],
            "LastPriceAt": ["2024-01-02"],
            "PctToTarget": [0.1],
            "EntryTimeET": ["09:30"],
            "Status": ["OPEN"],
            "HitDateET": [pd.NA],
            "Expiry": ["2024-02-01"],
            "BuyK": [1],
            "SellK": [2],
            "TP": [12],
            "Notes": [""],
        }
    )

    called = {}
    monkeypatch.setattr(history.st, "dataframe", lambda df_arg, *a, **k: called.setdefault("df", df_arg))
    monkeypatch.setattr(history.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(history.st, "info", lambda *a, **k: None)

    history.outcomes_summary(df)

    displayed = called.get("df")
    assert isinstance(displayed, Styler)
    assert list(displayed.data.columns) == [
        "Ticker",
        "EvalDate",
        "Price",
        "RelVol(TimeAdj63d)",
        "LastPrice",
        "LastPriceAt",
        "PctToTarget",
        "EntryTimeET",
        "Status",
        "HitDateET",
        "Expiry",
        "DTE",
        "BuyK",
        "SellK",
        "TP",
        "Notes",
    ]


def test_render_history_tab_shows_extended_columns(monkeypatch):
    df_last = pd.DataFrame(
        {
            "Ticker": ["AAA"],
            "EvalDate": ["2024-01-01"],
            "run_date": ["2024-01-02"],
            "Price": [1],
            "Change%": [0.05],
            "RelVol(TimeAdj63d)": [1.5],
            "LastPrice": [1.1],
            "LastPriceAt": ["2024-01-02"],
            "PctToTarget": [0.2],
            "EntryTimeET": ["09:30"],
            "BuyK": [1],
            "SellK": [2],
            "TP": [2],
            "Extra": [3],
        }
    )

    calls = {}
    monkeypatch.setattr(history.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(history.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(
        history.st, "dataframe", lambda df_arg, *a, **k: calls.setdefault("df", df_arg)
    )
    monkeypatch.setattr(history, "load_outcomes", lambda: pd.DataFrame())
    monkeypatch.setattr(history, "latest_trading_day_recs", lambda _df: (df_last, "2024-01-01"))
    monkeypatch.setattr(history, "outcomes_summary", lambda _df: None)

    history.render_history_tab()

    displayed = calls.get("df")
    assert isinstance(displayed, Styler)
    assert list(displayed.data.columns) == [
        "Ticker",
        "EvalDate",
        "run_date",
        "Price",
        "Change%",
        "RelVol(TimeAdj63d)",
        "LastPrice",
        "LastPriceAt",
        "PctToTarget",
        "EntryTimeET",
        "BuyK",
        "SellK",
        "TP",
    ]


def test_render_scanner_tab_shows_dataframe(monkeypatch):
    df = pd.DataFrame({"Ticker": ["AAA"], "Price": [1], "RelVol(TimeAdj63d)": [1], "TP": [2]})

    calls = {}
    monkeypatch.setattr(scan.st, "markdown", lambda *a, **k: None)
    monkeypatch.setattr(scan.st, "button", lambda *a, **k: False)
    monkeypatch.setattr(scan.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(scan.st, "dataframe", lambda df_arg, *a, **k: calls.setdefault("df", df_arg))
    monkeypatch.setattr(scan, "_render_why_buy_block", lambda df_arg: None)
    monkeypatch.setattr(scan.st, "table", lambda *a, **k: None)
    monkeypatch.setattr(scan.st, "caption", lambda *a, **k: None)

    @contextmanager
    def dummy_expander(*args, **kwargs):
        yield

    monkeypatch.setattr(scan.st, "expander", dummy_expander)
    monkeypatch.setattr(scan.st, "session_state", {"last_pass": df})

    scan.render_scanner_tab()

    assert isinstance(calls.get("df"), Styler)


def test_style_negatives_marks_negatives():
    df = pd.DataFrame({"PctChange": [1, -2]})
    styler = scan._style_negatives(df)
    html = styler.to_html()
    assert 'neg"' in html

