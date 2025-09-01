import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import utils.outcomes as outcomes  # noqa: E402

def _stub_history(ticker, start=None, end=None, auto_adjust=False):
    dates = pd.date_range('2020-01-01', periods=5, freq='D')
    highs = [8, 10, 12, 11, 9]
    return pd.DataFrame({'High': highs}, index=dates)

def test_evaluate_pending(monkeypatch):
    monkeypatch.setattr(outcomes, 'fetch_history', _stub_history)
    df = pd.DataFrame([
        {
            'Ticker': 'ABC',
            'EvalDate': '2020-01-01',
            'result_status': 'PENDING',
            'TP': 10,
            'OptExpiry': '2020-01-10',
        }
    ])
    out = outcomes.evaluate_outcomes(df)
    row = out.iloc[0]
    assert row['result_status'] == 'HIT'
    assert row['hit_time'] == '2020-01-02'
    assert row['hit_price'] == 10

def test_evaluate_history(monkeypatch):
    monkeypatch.setattr(outcomes, 'fetch_history', _stub_history)
    df = pd.DataFrame([
        {
            'Ticker': 'XYZ',
            'EvalDate': '2020-01-01',
            'WindowEnd': '2020-01-05',
            'TargetLevel': 15,
            'Outcome': 'PENDING',
        }
    ])
    out = outcomes.evaluate_outcomes(df)
    row = out.iloc[0]
    assert row['Outcome'] == 'NO'
    assert row['MaxHigh'] == 12
