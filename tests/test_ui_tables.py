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
import ui.table_utils as table_utils
import ui.table_render as table_render


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

    html_calls = []
    monkeypatch.setattr(
        history.st,
        "markdown",
        lambda html_arg, *a, **k: html_calls.append(html_arg),
    )
    monkeypatch.setattr(history.st, "caption", lambda *a, **k: None)
    monkeypatch.setattr(history.st, "info", lambda *a, **k: None)

    history.outcomes_summary(df)

    assert html_calls
    parsed = pd.read_html(html_calls[0], index_col=0)[0]
    assert list(parsed.columns) == [
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
    df_pass = pd.DataFrame(
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
            "HitDateET": [pd.NA],
            "Expiry": ["2024-02-01"],
            "DTE": [10],
            "BuyK": [1],
            "SellK": [2],
            "TP": [2],
            "Notes": [""],
            "Extra": [3],
        }
    )

    html_calls = []
    monkeypatch.setattr(history.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(history.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(
        history.st,
        "markdown",
        lambda html_arg, *a, **k: html_calls.append(html_arg),
    )
    monkeypatch.setattr(history.st, "session_state", {})

    @contextmanager
    def dummy_col():
        yield

    monkeypatch.setattr(history.st, "columns", lambda *a, **k: (dummy_col(), dummy_col()))

    monkeypatch.setattr(history, "load_pass_history", lambda: df_pass)
    monkeypatch.setattr(history, "latest_trading_day_recs", lambda _df: (df_pass, "2024-01-02"))

    history.render_history_tab()

    assert len(html_calls) == 2
    parsed = pd.read_html(html_calls[0], index_col=0)[0]
    assert list(parsed.columns) == [
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
        "HitDateET",
        "Expiry",
        "DTE",
        "BuyK",
        "SellK",
        "TP",
        "Notes",
        "Extra",
    ]


def test_render_history_tab_injects_row_select_once(monkeypatch):
    df_pass = pd.DataFrame(
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
            "HitDateET": [pd.NA],
            "Expiry": ["2024-02-01"],
            "DTE": [10],
            "BuyK": [1],
            "SellK": [2],
            "TP": [2],
            "Notes": [""],
        }
    )

    html_calls: list[str] = []
    monkeypatch.setattr(history.st, "subheader", lambda *a, **k: None)
    monkeypatch.setattr(history.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(history.st, "markdown", lambda html, *a, **k: html_calls.append(html))
    monkeypatch.setattr(history.st, "session_state", {})

    @contextmanager
    def dummy_col():
        yield

    monkeypatch.setattr(history.st, "columns", lambda *a, **k: (dummy_col(), dummy_col()))
    monkeypatch.setattr(history, "load_pass_history", lambda: df_pass)
    monkeypatch.setattr(history, "latest_trading_day_recs", lambda _df: (df_pass, "2024-01-02"))

    history._row_select_injected = False
    history.render_history_tab()

    assert len(html_calls) == 2
    total = sum(h.count("row-select-js") for h in html_calls)
    assert total == 1


def test_render_scanner_tab_shows_dataframe(monkeypatch):
    df = pd.DataFrame({"Ticker": ["AAA"], "Price": [1], "RelVol(TimeAdj63d)": [1], "TP": [2]})

    html_calls = []
    monkeypatch.setattr(scan.st, "markdown", lambda html_arg, *a, **k: html_calls.append(html_arg))
    monkeypatch.setattr(scan.st, "button", lambda *a, **k: False)
    monkeypatch.setattr(scan.st, "info", lambda *a, **k: None)
    monkeypatch.setattr(scan, "_render_why_buy_block", lambda df_arg: None)
    monkeypatch.setattr(scan.st, "caption", lambda *a, **k: None)

    @contextmanager
    def dummy_expander(*args, **kwargs):
        yield

    monkeypatch.setattr(scan.st, "expander", dummy_expander)
    monkeypatch.setattr(scan.st, "session_state", {"last_pass": df})

    scan.render_scanner_tab()
    table_html = next((h for h in html_calls if "<table" in h), None)
    assert table_html is not None
    parsed = pd.read_html(table_html, index_col=0)[0]
    assert list(parsed.columns) == ["Ticker", "Price", "RelVol(TimeAdj63d)", "TP"]


def test_style_negatives_marks_both_signs():
    df = pd.DataFrame({"PctChange": [1, -2, 0]})
    styler = table_utils._style_negatives(df)
    html = styler.to_html()
    assert 'neg"' in html
    assert 'pos"' in html


def test_render_table_injects_script_once(monkeypatch):
    df = pd.DataFrame({"Ticker": ["AAA"], "Price": [1]})
    monkeypatch.setattr(table_render, "_script_emitted", False)
    html1 = table_render.render_table(df)
    html2 = table_render.render_table(df)
    combined = html1 + html2
    assert combined.count('id="row-select-js"') == 1


def test_inject_row_select_js_does_not_reinject(monkeypatch):
    html = "<table></table>"
    monkeypatch.setattr(table_utils.st, "session_state", {})
    first = table_utils.inject_row_select_js(html)
    second = table_utils.inject_row_select_js(html)
    assert first.count("<script>") == 1
    assert second.count("<script>") == 0

