import pandas as pd


def test_backtest_wide_assembly():
    prices_df = pd.DataFrame(
        {
            "Ticker": ["AAA", "AAA", "BBB"],
            "Close": [1.0, 2.0, 3.0],
            "Open": [1.0, 2.0, 3.0],
            "Volume": [10, 20, 30],
        },
        index=[pd.Timestamp("2020-01-01")] * 3,
    )
    rows_before = len(prices_df)
    prices_df = prices_df.reset_index(names="date")
    prices_df = (
        prices_df.drop_duplicates(subset=["Ticker", "date"], keep="last")
        .set_index("date")
        .sort_index()
    )
    rows_after = len(prices_df)
    close_wide = prices_df.pivot_table(
        index=prices_df.index, columns="Ticker", values="Close", aggfunc="last"
    )
    open_wide = prices_df.pivot_table(
        index=prices_df.index, columns="Ticker", values="Open", aggfunc="last"
    )
    vol_wide = prices_df.pivot_table(
        index=prices_df.index, columns="Ticker", values="Volume", aggfunc="last"
    )
    assert rows_before == 3
    assert rows_after == 2
    assert close_wide.loc[pd.Timestamp("2020-01-01"), "AAA"] == 2.0
    assert close_wide.shape == (1, 2)
    assert open_wide.shape == (1, 2)
    assert vol_wide.shape == (1, 2)
