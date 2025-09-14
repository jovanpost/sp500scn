import pandas as pd
from utils import prices as up


def test_fetch_history_accepts_lowercase(monkeypatch):
    def fake_load_prices_cached(storage, tickers, start, end):
        df = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "date": [pd.Timestamp("2020-01-01")],
                "open": [1],
                "high": [1],
                "low": [1],
                "close": [1],
                "volume": [100],
            }
        )
        return df.set_index("date")

    monkeypatch.setattr(up, "load_prices_cached", fake_load_prices_cached)

    out = up.fetch_history("AAA")
    assert set(["Open", "High", "Low", "Close", "Adj Close", "Volume", "Ticker"]).issubset(out.columns)
    assert out.iloc[0]["Open"] == 1
