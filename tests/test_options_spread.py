import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.run_range import compute_spread_summary
import engine.options_spread as options_spread_module
from engine.options_spread import (
    OptionsSpreadConfig,
    black_scholes_call,
    black_scholes_put,
    build_vertical_spread,
    choose_vertical_call_legs,
    compute_contracts_and_costs,
    compute_vertical_spread_trade,
    evaluate_vertical_spread,
    estimate_vol,
    price_vertical_spread,
)


def test_estimate_vol_parkinson_matches_formula():
    close = pd.Series(np.linspace(100, 120, 30))
    high = close * 1.02
    low = close * 0.98
    lookback = 21

    sigma = estimate_vol(close, high, low, lookback=lookback, method="parkinson")

    tail_high = high.tail(lookback).to_numpy()
    tail_low = low.tail(lookback).to_numpy()
    log_ratio = np.log(tail_high / tail_low)
    expected = math.sqrt((log_ratio**2).sum() / (4 * len(log_ratio) * math.log(2.0))) * math.sqrt(252.0)

    assert sigma == pytest.approx(expected, rel=1e-6)


def test_estimate_vol_close_fallback():
    close = pd.Series(np.linspace(95, 110, 25))
    high = pd.Series([100.0] * 25)
    low = pd.Series([0.0] * 25)  # invalid lows force fallback

    sigma = estimate_vol(close, high, low, lookback=21, method="parkinson")

    log_returns = np.diff(np.log(close.tail(22).to_numpy()))
    expected = np.std(log_returns, ddof=1) * math.sqrt(252.0)

    assert sigma == pytest.approx(expected, rel=1e-6)


def test_estimate_vol_atr_fallback():
    close = pd.Series([100.0] * 30)
    high = pd.Series([100.0] * 30)
    low = pd.Series([100.0] * 30)

    sigma = estimate_vol(close, high, low, lookback=21, method="parkinson", atr=2.5)
    expected = (2.5 / 100.0) * math.sqrt(252.0)

    assert sigma == pytest.approx(expected, rel=1e-6)


def test_black_scholes_call_put_parity():
    spot = 100.0
    strike = 95.0
    time = 0.25
    sigma = 0.2
    r = 0.01
    q = 0.02

    call = black_scholes_call(spot, strike, time, sigma, r, q)
    put = black_scholes_put(spot, strike, time, sigma, r, q)

    lhs = call - put
    rhs = spot * math.exp(-q * time) - strike * math.exp(-r * time)

    assert lhs == pytest.approx(rhs, rel=1e-6)


def test_black_scholes_zero_time_intrinsic():
    assert black_scholes_call(105.0, 100.0, 0.0, 0.2) == pytest.approx(5.0)
    assert black_scholes_put(95.0, 100.0, 0.0, 0.2) == pytest.approx(5.0)


def test_choose_vertical_call_legs_basic():
    lower, upper, meta = choose_vertical_call_legs(
        tp_abs_target=20.0, strike_tick=0.5, tp_anchor_offset_ticks=1, min_width_ticks=1
    )

    assert upper == pytest.approx(19.5)
    assert lower == pytest.approx(19.0)
    assert meta["k2_target"] == pytest.approx(19.5)


def test_choose_vertical_call_legs_invalid_target():
    lower, upper, meta = choose_vertical_call_legs(
        tp_abs_target=-5.0, strike_tick=1.0, tp_anchor_offset_ticks=1, min_width_ticks=1
    )

    assert lower is None and upper is None
    assert meta["opt_reason"] == "tp_target_invalid"


def test_compute_contracts_and_costs_budget_math():
    contracts, cash = compute_contracts_and_costs(
        debit_entry=7.94, budget_per_trade=794.0, fees_per_contract=0.0
    )
    assert contracts == 1
    assert cash == pytest.approx(794.0)

    contracts_small, cash_small = compute_contracts_and_costs(
        debit_entry=7.94, budget_per_trade=793.99, fees_per_contract=0.0
    )
    assert contracts_small == 0
    assert cash_small == pytest.approx(0.0)

    contracts_with_fees, cash_with_fees = compute_contracts_and_costs(
        debit_entry=1.23, budget_per_trade=500.0, fees_per_contract=0.65
    )
    assert contracts_with_fees == 4
    assert cash_with_fees == pytest.approx(497.2, rel=1e-6)


def test_build_vertical_spread_affordable():
    cfg = OptionsSpreadConfig(
        budget_per_trade=1000.0,
        fees_per_contract=0.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )

    spread, meta = build_vertical_spread(
        spot=100.0, direction="up", sigma=0.2, config=cfg, tp_abs_target=110.0
    )
    assert spread is not None
    assert meta.get("opt_reason") == ""
    assert meta.get("opt_structure") == "CALL_VERTICAL_DEBIT"
    assert spread.structure == "bull_call"
    assert spread.lower_strike == pytest.approx(108.0, rel=1e-6)
    assert spread.upper_strike == pytest.approx(109.0, rel=1e-6)
    expected_width_frac = (spread.upper_strike - spread.lower_strike) / 100.0
    assert spread.width_frac == pytest.approx(expected_width_frac, rel=1e-6)
    assert spread.contracts >= 1
    assert spread.cash_outlay <= cfg.budget_per_trade + 1e-6


def test_build_vertical_spread_affordability_shift():
    cfg = OptionsSpreadConfig(
        budget_per_trade=150.0,
        fees_per_contract=0.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
        tp_anchor_mode=False,
    )

    spread, meta = build_vertical_spread(spot=100.0, direction="up", sigma=0.8, config=cfg)
    assert spread is not None
    assert meta.get("opt_reason") == ""
    assert spread.lower_strike > 100.0  # shifted OTM to fit budget
    assert spread.upper_strike - spread.lower_strike == pytest.approx(5.0, rel=1e-6)
    assert spread.width_frac == pytest.approx((spread.upper_strike - spread.lower_strike) / 100.0, rel=1e-6)
    assert spread.cash_outlay <= cfg.budget_per_trade + 1e-6


def test_exit_vertical_spread_intrinsic_cap():
    cfg = OptionsSpreadConfig(
        budget_per_trade=1000.0,
        fees_per_contract=0.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )
    spread, _ = build_vertical_spread(
        spot=100.0, direction="up", sigma=0.2, config=cfg, tp_abs_target=110.0
    )
    assert spread is not None

    width = spread.upper_strike - spread.lower_strike
    outcome = evaluate_vertical_spread(
        spread,
        S_exit=spread.upper_strike + 1.0,
        days_to_expiry=0,
        sigma_exit=spread.sigma_entry,
        config=cfg,
    )

    assert outcome.debit_exit == pytest.approx(width, rel=1e-9)
    assert outcome.revenue == pytest.approx(width * spread.contracts * 100.0, rel=1e-9)
    assert outcome.pnl == pytest.approx(outcome.revenue - spread.cash_outlay, rel=1e-9)


def test_exit_vertical_spread_otm_loss():
    cfg = OptionsSpreadConfig(
        budget_per_trade=1000.0,
        fees_per_contract=0.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )
    spread, _ = build_vertical_spread(
        spot=100.0, direction="up", sigma=0.25, config=cfg, tp_abs_target=110.0
    )
    assert spread is not None

    outcome = evaluate_vertical_spread(
        spread,
        S_exit=spread.lower_strike - 5.0,
        days_to_expiry=0,
        sigma_exit=spread.sigma_entry,
        config=cfg,
    )

    assert outcome.debit_exit == pytest.approx(0.0, abs=1e-9)
    assert outcome.revenue == pytest.approx(0.0, abs=1e-9)
    assert outcome.pnl == pytest.approx(-spread.cash_outlay, rel=1e-9)


def test_compute_vertical_spread_trade_matches_bs_exit():
    dates = pd.bdate_range("2023-01-02", periods=60)
    close = pd.Series([100.0] * len(dates), index=dates)
    high = pd.Series([102.0] * len(dates), index=dates)
    low = pd.Series([98.0] * len(dates), index=dates)
    prices = pd.DataFrame({"close": close, "high": high, "low": low})

    entry_ts = dates[20]
    exit_ts = dates[25]
    exit_price = 103.0

    cfg = OptionsSpreadConfig(
        budget_per_trade=1000.0,
        fees_per_contract=0.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
        vol_multiplier=1.0,
    )

    result = compute_vertical_spread_trade(
        prices=prices,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
        entry_price=100.0,
        exit_price=exit_price,
        direction="up",
        config=cfg,
        atr_value=None,
        exit_reason="tp",
        tp_abs_target=105.0,
    )

    assert result["contracts"] >= 1
    assert result["debit_entry"] > 0
    assert result["width_frac"] == pytest.approx((result["K2"] - result["K1"]) / 100.0, rel=1e-6)
    assert result["opt_structure"] == "CALL_VERTICAL_DEBIT"
    assert result["opt_reason"] == ""

    days_to_exit = int(result["T_exit_days"])
    if days_to_exit > 0:
        expected_exit = price_vertical_spread(
            "bull_call",
            exit_price,
            result["K1"],
            result["K2"],
            max(days_to_exit / 365.0, 1.0 / 3650.0),
            result["sigma_exit"],
            cfg.risk_free_rate,
            cfg.dividend_yield,
        )
    else:
        expected_exit = min(max(exit_price - result["K1"], 0.0), result["K2"] - result["K1"])

    assert result["debit_exit"] == pytest.approx(expected_exit, abs=0.05)
    assert result["revenue"] == pytest.approx(result["debit_exit"] * result["contracts"] * 100.0, rel=1e-9)


def test_build_vertical_spread_rejects_debit_gt_width(monkeypatch):
    cfg = OptionsSpreadConfig(
        budget_per_trade=1000.0,
        fees_per_contract=0.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )

    monkeypatch.setattr(options_spread_module, "price_vertical_spread", lambda *args, **kwargs: 1.5)
    spread, meta = build_vertical_spread(
        spot=100.0, direction="up", sigma=0.2, config=cfg, tp_abs_target=105.0
    )

    assert spread is None
    assert meta.get("opt_reason") == "invalid_debit_gt_width"


def test_compute_vertical_spread_trade_budget_skip():
    dates = pd.bdate_range("2023-01-02", periods=40)
    close = pd.Series([100.0] * len(dates), index=dates)
    high = pd.Series([102.0] * len(dates), index=dates)
    low = pd.Series([98.0] * len(dates), index=dates)
    prices = pd.DataFrame({"close": close, "high": high, "low": low})

    cfg = OptionsSpreadConfig(
        budget_per_trade=10.0,
        fees_per_contract=0.0,
        risk_free_rate=0.0,
        dividend_yield=0.0,
    )

    result = compute_vertical_spread_trade(
        prices=prices,
        entry_ts=dates[10],
        exit_ts=dates[12],
        entry_price=100.0,
        exit_price=101.0,
        direction="up",
        config=cfg,
        atr_value=None,
        exit_reason="tp",
        tp_abs_target=105.0,
    )

    assert result["contracts"] == 0
    assert result["opt_reason"] == "insufficient_budget"
    assert math.isnan(result["cash_outlay"]) or result["cash_outlay"] == 0.0 or result["cash_outlay"] == pytest.approx(0.0)


def test_compute_spread_summary_includes_dollar_fields():
    trades_df = pd.DataFrame(
        {
            "contracts": [1, 2, 0],
            "cash_outlay": [900.0, 800.0, 500.0],
            "revenue": [1200.0, 0.0, 0.0],
            "fees_exit": [2.6, 2.6, 0.0],
            "pnl_dollars": [297.4, -802.6, 0.0],
        }
    )

    summary = compute_spread_summary(trades_df)

    assert summary["trades_executed_spread"] == 2
    assert summary["invested_dollars"] == pytest.approx(1700.0)
    assert summary["gross_revenue_dollars"] == pytest.approx(1200.0)
    assert summary["net_pnl_spread"] == pytest.approx(-505.2)
    assert summary["end_value_dollars"] == pytest.approx(1194.8)
    assert summary["avg_cost_per_trade"] == pytest.approx(850.0)
    assert summary["dollar_summary_str"] == (
        "2 trades — $1,700 invested → $1,195 end value ⇒ $-505 net loss"
    )


def test_compute_spread_summary_basic():
    df = pd.DataFrame(
        {
            "contracts": [1, 2, 0],
            "pnl_dollars": [150.0, -50.0, float("nan")],
        }
    )

    summary = compute_spread_summary(df)

    assert summary["trades_spread"] == 2
    assert summary["wins_spread"] == 1
    assert summary["losses_spread"] == 1
    assert summary["win_rate_spread"] == pytest.approx(0.5)
    assert summary["gross_profit_spread"] == pytest.approx(150.0)
    assert summary["gross_loss_spread"] == pytest.approx(-50.0)
    assert summary["net_pnl_spread"] == pytest.approx(100.0)
    assert summary["avg_pnl_per_trade"] == pytest.approx(50.0)
    assert summary["max_drawdown_spread"] == pytest.approx(-50.0)
