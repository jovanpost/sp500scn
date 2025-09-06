from utils import market_calendar as mc


def test_trading_days_between_excludes_weekend():
    days = mc.trading_days_between("2024-07-05", "2024-07-08")
    assert [d.strftime("%Y-%m-%d") for d in days] == ["2024-07-05", "2024-07-08"]


def test_is_trading_day_handles_weekend():
    assert mc.is_trading_day("2024-07-05")
    assert not mc.is_trading_day("2024-07-06")
