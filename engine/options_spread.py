from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


_ANNUALIZATION_FACTOR = math.sqrt(252.0)
_EPSILON_TIME = 1.0 / 3650.0


def floor_to_tick(value: float, tick: float) -> float:
    """Floor a value to the nearest lower multiple of ``tick``."""

    if tick <= 0 or not math.isfinite(value):
        return float("nan")
    floored = math.floor(value / tick) * tick
    return round(floored, 6)


def choose_vertical_call_legs(
    tp_abs_target: float,
    strike_tick: float,
    tp_anchor_offset_ticks: int = 1,
    min_width_ticks: int = 1,
) -> tuple[float | None, float | None, dict]:
    """Select strikes for a call debit spread anchored to the TP target."""

    meta: dict[str, float] = {}
    if strike_tick <= 0 or not math.isfinite(strike_tick):
        return None, None, {"opt_reason": "invalid_strike_tick", **meta}

    offset_ticks = max(int(tp_anchor_offset_ticks), 1)
    width_ticks = max(int(min_width_ticks), 1)

    if not math.isfinite(tp_abs_target) or tp_abs_target <= 0:
        return None, None, {"opt_reason": "tp_target_invalid", **meta}

    k2_target = tp_abs_target - offset_ticks * strike_tick
    meta["k2_target"] = k2_target
    K2 = floor_to_tick(k2_target, strike_tick)
    if not math.isfinite(K2) or K2 <= 0:
        return None, None, {"opt_reason": "invalid_leg_nonpositive", **meta}

    K1 = round(K2 - width_ticks * strike_tick, 6)
    if K1 <= 0:
        return None, None, {"opt_reason": "invalid_leg_nonpositive", **meta}

    if not (K2 < tp_abs_target):
        return None, None, {"opt_reason": "k2_not_below_tp", **meta}
    if not (K1 < K2):
        return None, None, {"opt_reason": "k1_not_below_k2", **meta}

    width_ratio = (K2 - K1) / strike_tick
    if not math.isfinite(width_ratio) or abs(width_ratio - round(width_ratio)) > 1e-6:
        return None, None, {"opt_reason": "width_not_tick_multiple", **meta}

    return float(K1), float(K2), meta


def compute_contracts_and_costs(
    debit_entry: float,
    budget_per_trade: float,
    fees_per_contract: float,
    *,
    legs: int = 2,
) -> tuple[int, float]:
    """Return contracts purchasable and total cash outlay including entry fees."""

    if debit_entry <= 0 or not math.isfinite(debit_entry):
        return 0, 0.0

    entry_fees_per_contract = max(legs, 0) * max(float(fees_per_contract), 0.0)
    cost_per_contract_entry = debit_entry * 100.0 + entry_fees_per_contract
    if cost_per_contract_entry <= 0:
        return 0, 0.0

    affordable = float(budget_per_trade)
    if not math.isfinite(affordable) or affordable <= 0:
        return 0, 0.0

    contracts = int(affordable // cost_per_contract_entry)
    if contracts <= 0:
        return 0, 0.0

    cash_outlay = contracts * debit_entry * 100.0 + contracts * entry_fees_per_contract
    return contracts, float(cash_outlay)


@dataclass
class OptionsSpreadConfig:
    enabled: bool = True
    kind: str = "vertical_debit"
    budget_per_trade: float = 1000.0
    expiry_days: int = 30
    width_frac: float = 0.05
    width_abs: float | None = None
    vol_lookback_days: int = 21
    vol_method: str = "parkinson"
    vol_multiplier: float = 1.0
    use_exit_vol_recalc: bool = False
    risk_free_rate: float = 0.05
    dividend_yield: float = 0.0
    max_otm_shift_pct: float = 20.0
    fees_per_contract: float = 0.65
    strike_tick: float = 1.0
    tp_anchor_mode: bool = True
    tp_anchor_offset_ticks: int = 1
    min_width_ticks: int = 1
    enforce_debit_le_width: bool = True

    @classmethod
    def from_params(
        cls, params: Optional[dict | "OptionsSpreadConfig"]
    ) -> "OptionsSpreadConfig":
        if isinstance(params, cls):
            return params

        cfg = cls()
        if not isinstance(params, dict):
            return cfg

        updates: Dict[str, object] = {}
        params_copy = dict(params)
        if "width_frac" not in params_copy and "width_pct" in params_copy:
            width_val = params_copy.get("width_pct")
            if isinstance(width_val, np.generic):
                width_val = width_val.item()
            try:
                width_float = float(width_val)
            except (TypeError, ValueError):
                width_float = None
            if width_float is not None:
                if width_float > 1.0:
                    width_float = width_float / 100.0
                params_copy["width_frac"] = width_float

        for f in fields(cls):
            if f.name not in params_copy or params_copy[f.name] is None:
                continue
            val = params_copy[f.name]
            if isinstance(val, np.generic):
                val = val.item()
            if isinstance(val, str):
                val = val.strip()
            if f.name in {"kind", "vol_method"} and isinstance(val, str):
                val = val.lower()
            if f.type is bool:
                updates[f.name] = bool(val)
            else:
                updates[f.name] = val

        if updates:
            return replace(cfg, **updates)
        return cfg

    def empty_result(self) -> dict:
        return {
            "opt_structure": "",
            "K1": float("nan"),
            "K2": float("nan"),
            "width_frac": float("nan"),
            "width_pct": float("nan"),
            "T_entry_days": float("nan"),
            "sigma_entry": float("nan"),
            "debit_entry": float("nan"),
            "contracts": 0,
            "cash_outlay": 0.0,
            "fees_entry": 0.0,
            "S_exit": float("nan"),
            "T_exit_days": float("nan"),
            "sigma_exit": float("nan"),
            "debit_exit": float("nan"),
            "revenue": 0.0,
            "fees_exit": 0.0,
            "pnl_dollars": 0.0,
            "win": pd.NA,
            "opt_reason": "",
        }


@dataclass
class VerticalSpread:
    structure: str
    lower_strike: float
    upper_strike: float
    width_frac: float
    expiry_days: int
    sigma_entry: float
    debit_entry: float
    contracts: int
    cash_outlay: float
    fees_entry: float

    @property
    def spread_width(self) -> float:
        return float(self.upper_strike - self.lower_strike)


@dataclass
class SpreadOutcome:
    S_exit: float
    T_exit_days: int
    sigma_exit: float
    debit_exit: float
    revenue: float
    fees_exit: float
    pnl: float
    win: bool


def _as_series(values: Iterable[float] | pd.Series, lookback: int) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.dropna().astype(float).tail(int(lookback))
    return pd.Series(list(values), dtype=float).dropna().tail(int(lookback))


def estimate_vol(
    close: Iterable[float] | pd.Series,
    high: Iterable[float] | pd.Series,
    low: Iterable[float] | pd.Series,
    lookback: int,
    *,
    method: str = "parkinson",
    atr: float | Iterable[float] | None = None,
    vol_multiplier: float = 1.0,
) -> float:
    """Estimate annualized volatility using OHLC data with fallbacks."""

    method = (method or "parkinson").lower()
    order: tuple[str, ...]
    if method == "parkinson":
        order = ("parkinson", "close", "atr")
    elif method == "close":
        order = ("close", "parkinson", "atr")
    else:
        order = ("atr", "parkinson", "close")

    close_s = _as_series(close, lookback + 1)
    high_s = _as_series(high, lookback)
    low_s = _as_series(low, lookback)
    atr_s: Optional[pd.Series] = None
    if atr is not None:
        if isinstance(atr, pd.Series):
            atr_s = atr.dropna().astype(float)
        else:
            atr_s = pd.Series([float(atr)], dtype=float)

    for mode in order:
        if mode == "parkinson":
            sigma = _estimate_parkinson(high_s, low_s)
        elif mode == "close":
            sigma = _estimate_close(close_s)
        else:  # ATR fallback
            sigma = _estimate_atr(close_s, atr_s)
        if sigma is not None and not math.isnan(sigma) and sigma > 0:
            return float(sigma) * float(vol_multiplier)
    return float("nan")


def _estimate_parkinson(high: pd.Series, low: pd.Series) -> float | None:
    if high.empty or low.empty:
        return None
    hl = pd.concat([high, low], axis=1)
    hl = hl[(hl.iloc[:, 0] > 0) & (hl.iloc[:, 1] > 0)]
    if hl.empty:
        return None
    log_ratio = np.log(hl.iloc[:, 0].to_numpy() / hl.iloc[:, 1].to_numpy())
    log_ratio = log_ratio[np.isfinite(log_ratio)]
    n = len(log_ratio)
    if n == 0:
        return None
    variance = (log_ratio**2).sum() / (4.0 * n * math.log(2.0))
    return math.sqrt(max(variance, 0.0)) * _ANNUALIZATION_FACTOR


def _estimate_close(close: pd.Series) -> float | None:
    if len(close) < 2:
        return None
    log_returns = np.diff(np.log(close.to_numpy()))
    if len(log_returns) == 0:
        return None
    std = np.nanstd(log_returns, ddof=1 if len(log_returns) > 1 else 0)
    if not np.isfinite(std):
        return None
    return float(std) * _ANNUALIZATION_FACTOR


def _estimate_atr(close: pd.Series, atr: Optional[pd.Series]) -> float | None:
    if atr is None or atr.empty or close.empty:
        return None
    atr_val = float(atr.dropna().iloc[-1]) if not atr.dropna().empty else float("nan")
    if not math.isfinite(atr_val) or atr_val <= 0:
        return None
    spot = float(close.dropna().iloc[-1]) if not close.dropna().empty else float("nan")
    if not math.isfinite(spot) or spot <= 0:
        return None
    return (atr_val / spot) * _ANNUALIZATION_FACTOR


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(
    spot: float,
    strike: float,
    time: float,
    sigma: float,
    r: float = 0.05,
    q: float = 0.0,
) -> float:
    if time <= 0 or sigma <= 0:
        return max(spot - strike, 0.0)
    if spot <= 0 or strike <= 0:
        return 0.0
    sqrt_t = math.sqrt(time)
    sigma = max(sigma, 1e-9)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma**2) * time) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return spot * math.exp(-q * time) * _norm_cdf(d1) - strike * math.exp(-r * time) * _norm_cdf(d2)


def black_scholes_put(
    spot: float,
    strike: float,
    time: float,
    sigma: float,
    r: float = 0.05,
    q: float = 0.0,
) -> float:
    if time <= 0 or sigma <= 0:
        return max(strike - spot, 0.0)
    if spot <= 0 or strike <= 0:
        return 0.0
    sqrt_t = math.sqrt(time)
    sigma = max(sigma, 1e-9)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma**2) * time) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return strike * math.exp(-r * time) * _norm_cdf(-d2) - spot * math.exp(-q * time) * _norm_cdf(-d1)


def _round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return float(price)
    return round(price / tick) * tick


def price_vertical_spread(
    structure: str,
    spot: float,
    lower: float,
    upper: float,
    time: float,
    sigma: float,
    r: float,
    q: float,
) -> float:
    structure = structure.lower()
    if time <= 0:
        return intrinsic_value(structure, spot, lower, upper)

    sigma = max(float(sigma), 1e-8)
    time = max(float(time), _EPSILON_TIME)

    if structure == "bull_call":
        return black_scholes_call(spot, lower, time, sigma, r, q) - black_scholes_call(
            spot, upper, time, sigma, r, q
        )
    if structure == "bear_put":
        return black_scholes_put(spot, upper, time, sigma, r, q) - black_scholes_put(
            spot, lower, time, sigma, r, q
        )
    raise ValueError(f"Unsupported spread structure: {structure}")


def intrinsic_value(structure: str, spot: float, lower: float, upper: float) -> float:
    width = max(upper - lower, 0.0)
    if width <= 0:
        return 0.0
    if structure == "bull_call":
        intrinsic = max(spot - lower, 0.0) - max(spot - upper, 0.0)
    elif structure == "bear_put":
        intrinsic = max(upper - spot, 0.0) - max(lower - spot, 0.0)
    else:
        raise ValueError(f"Unsupported spread structure: {structure}")
    intrinsic = max(min(intrinsic, width), 0.0)
    return intrinsic


def build_vertical_spread(
    spot: float,
    direction: str,
    sigma: float,
    config: OptionsSpreadConfig,
    tp_abs_target: float | None = None,
) -> tuple[VerticalSpread | None, dict]:
    meta: dict[str, object] = {}
    if not config.enabled or config.kind != "vertical_debit":
        meta["opt_reason"] = "options_disabled"
        return None, meta
    if not math.isfinite(spot) or spot <= 0 or not math.isfinite(sigma) or sigma <= 0:
        meta["opt_reason"] = "invalid_underlying"
        return None, meta

    tick = float(config.strike_tick or 1.0)
    structure = "bull_call" if direction.lower() != "down" else "bear_put"
    meta["opt_structure"] = (
        "CALL_VERTICAL_DEBIT" if structure == "bull_call" else "PUT_VERTICAL_DEBIT"
    )

    time_years = max(float(config.expiry_days) / 365.0, _EPSILON_TIME)
    r = float(config.risk_free_rate)
    q = float(config.dividend_yield)

    if config.tp_anchor_mode and structure == "bull_call":
        tp_value = float(tp_abs_target) if tp_abs_target is not None else float("nan")
        lower, upper, anchor_meta = choose_vertical_call_legs(
            tp_value,
            strike_tick=tick,
            tp_anchor_offset_ticks=int(config.tp_anchor_offset_ticks),
            min_width_ticks=int(config.min_width_ticks),
        )
        meta.update(anchor_meta)
        if lower is None or upper is None:
            meta.setdefault("opt_reason", anchor_meta.get("opt_reason", "tp_anchor_failed"))
            return None, meta

        spread_width = float(upper - lower)
        width_frac = spread_width / spot if spot else float("nan")
        debit = price_vertical_spread(structure, spot, lower, upper, time_years, sigma, r, q)
        meta.update(
            {
                "K1": float(lower),
                "K2": float(upper),
                "width_frac": float(width_frac),
                "debit_entry": float(debit),
            }
        )

        if not math.isfinite(debit) or debit <= 0:
            meta["opt_reason"] = "pricing_failed"
            return None, meta

        width_dollars = spread_width * 100.0
        if config.enforce_debit_le_width and (debit * 100.0 > width_dollars + 1e-6):
            meta.update({"opt_reason": "invalid_debit_gt_width", "width_dollars": width_dollars})
            return None, meta

        contracts, cash_outlay = compute_contracts_and_costs(
            debit, config.budget_per_trade, config.fees_per_contract, legs=2
        )
        if contracts <= 0:
            meta["opt_reason"] = "insufficient_budget"
            return None, meta

        fees_entry = contracts * 2.0 * float(config.fees_per_contract)
        spread = VerticalSpread(
            structure=structure,
            lower_strike=float(lower),
            upper_strike=float(upper),
            width_frac=float(width_frac),
            expiry_days=int(config.expiry_days),
            sigma_entry=float(sigma),
            debit_entry=float(debit),
            contracts=int(contracts),
            cash_outlay=float(cash_outlay),
            fees_entry=float(fees_entry),
        )
        meta["opt_reason"] = ""
        return spread, meta

    atm = _round_to_tick(spot, tick)
    if structure == "bull_call":
        lower = atm
        if config.width_abs is not None and config.width_abs > 0:
            upper_target = lower + float(config.width_abs)
        else:
            width_frac = float(config.width_frac or 0.0)
            if width_frac <= 0:
                width_frac = 0.05
            upper_target = lower * (1.0 + width_frac)
        upper = _round_to_tick(upper_target, tick)
        if upper <= lower:
            upper = lower + tick
    else:
        upper = atm
        if config.width_abs is not None and config.width_abs > 0:
            lower_target = upper - float(config.width_abs)
        else:
            width_frac = float(config.width_frac or 0.0)
            if width_frac <= 0:
                width_frac = 0.05
            lower_target = upper * (1.0 - width_frac)
        lower = _round_to_tick(lower_target, tick)
        if lower >= upper:
            lower = max(upper - tick, tick)

    base_lower = float(lower)
    base_upper = float(upper)
    last_meta: dict[str, object] = {}

    attempts = [0.0]
    attempts.extend(np.arange(1.0, float(config.max_otm_shift_pct) + 1.0, 1.0))

    for pct in attempts:
        if pct > 0:
            shift = pct / 100.0
            if structure == "bull_call":
                lower = _round_to_tick(base_lower * (1.0 + shift), tick)
                upper = _round_to_tick(lower + (base_upper - base_lower), tick)
                if upper <= lower:
                    upper = lower + tick
            else:
                upper = _round_to_tick(base_upper * (1.0 - shift), tick)
                lower = _round_to_tick(upper - (base_upper - base_lower), tick)
                if lower >= upper:
                    lower = max(upper - tick, tick)

        spread_width = float(upper - lower)
        if spread_width <= 0:
            continue

        width_frac = spread_width / spot if spot else float("nan")
        debit = price_vertical_spread(structure, spot, lower, upper, time_years, sigma, r, q)
        attempt_meta = {
            "K1": float(lower),
            "K2": float(upper),
            "width_frac": float(width_frac),
            "debit_entry": float(debit),
        }

        if not math.isfinite(debit) or debit <= 0:
            last_meta = attempt_meta
            continue

        width_dollars = spread_width * 100.0
        if config.enforce_debit_le_width and (debit * 100.0 > width_dollars + 1e-6):
            attempt_meta.update({"opt_reason": "invalid_debit_gt_width", "width_dollars": width_dollars})
            meta.update(attempt_meta)
            return None, meta

        contracts, cash_outlay = compute_contracts_and_costs(
            debit, config.budget_per_trade, config.fees_per_contract, legs=2
        )
        if contracts <= 0:
            last_meta = attempt_meta
            continue

        fees_entry = contracts * 2.0 * float(config.fees_per_contract)
        spread = VerticalSpread(
            structure=structure,
            lower_strike=float(lower),
            upper_strike=float(upper),
            width_frac=float(width_frac),
            expiry_days=int(config.expiry_days),
            sigma_entry=float(sigma),
            debit_entry=float(debit),
            contracts=int(contracts),
            cash_outlay=float(cash_outlay),
            fees_entry=float(fees_entry),
        )
        attempt_meta["opt_reason"] = ""
        meta.update(attempt_meta)
        return spread, meta

    failure_meta = {"opt_reason": "insufficient_budget"}
    failure_meta.update(last_meta)
    meta.update(failure_meta)
    return None, meta


def evaluate_vertical_spread(
    spread: VerticalSpread,
    S_exit: float,
    days_to_expiry: int,
    sigma_exit: float,
    config: OptionsSpreadConfig,
) -> SpreadOutcome:
    days_to_expiry = int(days_to_expiry)
    if days_to_expiry <= 0:
        debit_exit = intrinsic_value(spread.structure, S_exit, spread.lower_strike, spread.upper_strike)
        sigma_used = float(sigma_exit)
    else:
        T = max(days_to_expiry / 365.0, _EPSILON_TIME)
        sigma_used = max(float(sigma_exit), 1e-8)
        debit_exit = price_vertical_spread(
            spread.structure,
            S_exit,
            spread.lower_strike,
            spread.upper_strike,
            T,
            sigma_used,
            float(config.risk_free_rate),
            float(config.dividend_yield),
        )

    contracts = spread.contracts
    revenue = contracts * debit_exit * 100.0
    fees_exit = contracts * 2.0 * float(config.fees_per_contract)
    pnl = revenue - spread.cash_outlay - fees_exit
    win = pnl > 0

    return SpreadOutcome(
        S_exit=float(S_exit),
        T_exit_days=max(days_to_expiry, 0),
        sigma_exit=float(sigma_used),
        debit_exit=float(debit_exit),
        revenue=float(revenue),
        fees_exit=float(fees_exit),
        pnl=float(pnl),
        win=bool(win),
    )


def compute_vertical_spread_trade(
    prices: pd.DataFrame,
    entry_ts: pd.Timestamp,
    exit_ts: pd.Timestamp,
    entry_price: float,
    exit_price: float,
    direction: str,
    config: OptionsSpreadConfig,
    atr_value: float | Iterable[float] | None = None,
    exit_reason: str | None = None,
    tp_abs_target: float | None = None,
) -> dict:
    result = config.empty_result()
    if not config.enabled or config.kind != "vertical_debit":
        result["opt_reason"] = "options_disabled"
        return result

    if (
        entry_price is None
        or exit_price is None
        or not math.isfinite(entry_price)
        or not math.isfinite(exit_price)
        or entry_price <= 0
        or exit_price <= 0
    ):
        result["opt_reason"] = "invalid_price_data"
        return result

    if not isinstance(prices.index, pd.DatetimeIndex):
        result["opt_reason"] = "invalid_price_index"
        return result

    entry_ts = pd.Timestamp(entry_ts).tz_localize(None)
    exit_ts = pd.Timestamp(exit_ts).tz_localize(None)
    if pd.isna(exit_ts):
        return result
    prices = prices.sort_index()
    prices.index = pd.to_datetime(prices.index).tz_localize(None)

    hist = prices.loc[prices.index < entry_ts]
    if hist.empty:
        result["opt_reason"] = "insufficient_history"
        return result

    lookback = max(int(config.vol_lookback_days), 1)
    sigma_entry = estimate_vol(
        hist["close"],
        hist["high"],
        hist["low"],
        lookback,
        method=config.vol_method,
        atr=atr_value,
        vol_multiplier=config.vol_multiplier,
    )
    if not math.isfinite(sigma_entry) or sigma_entry <= 0:
        result["opt_reason"] = "vol_estimate_invalid"
        return result

    spread, meta = build_vertical_spread(
        entry_price,
        direction,
        sigma_entry,
        config,
        tp_abs_target=tp_abs_target,
    )

    opt_structure = meta.get("opt_structure")
    if opt_structure:
        result["opt_structure"] = str(opt_structure)

    if meta:
        for key in ("K1", "K2", "width_frac", "debit_entry"):
            if key in meta and meta[key] is not None:
                try:
                    result[key] = float(meta[key])
                except (TypeError, ValueError):
                    continue
        if "width_frac" in meta and meta.get("width_frac") is not None:
            try:
                result["width_pct"] = float(meta["width_frac"]) * 100.0
            except (TypeError, ValueError):
                pass

    if spread is None or spread.contracts <= 0:
        result["opt_reason"] = str(meta.get("opt_reason", "spread_unavailable"))
        result["sigma_entry"] = float(sigma_entry)
        result["T_entry_days"] = float(config.expiry_days)
        return result

    expiry_ts = entry_ts + pd.Timedelta(days=int(config.expiry_days))
    days_to_expiry = max((expiry_ts - exit_ts).days, 0)

    sigma_exit = spread.sigma_entry
    if config.use_exit_vol_recalc and days_to_expiry > 0:
        hist_exit = prices.loc[prices.index <= exit_ts]
        sigma_new = estimate_vol(
            hist_exit["close"],
            hist_exit["high"],
            hist_exit["low"],
            lookback,
            method=config.vol_method,
            atr=atr_value,
            vol_multiplier=config.vol_multiplier,
        )
        if math.isfinite(sigma_new) and sigma_new > 0:
            sigma_exit = float(sigma_new)

    # Ensure TP exits use touched target when provided
    if isinstance(exit_reason, str) and exit_reason.lower() == "tp":
        S_exit = float(exit_price)
    else:
        S_exit = float(exit_price)

    outcome = evaluate_vertical_spread(spread, S_exit, days_to_expiry, sigma_exit, config)

    result.update(
        {
            "K1": float(spread.lower_strike),
            "K2": float(spread.upper_strike),
            "width_frac": float(spread.width_frac),
            "width_pct": float(spread.width_frac) * 100.0,
            "T_entry_days": float(spread.expiry_days),
            "sigma_entry": float(spread.sigma_entry),
            "debit_entry": float(spread.debit_entry),
            "contracts": int(spread.contracts),
            "cash_outlay": float(spread.cash_outlay),
            "fees_entry": float(spread.fees_entry),
            "S_exit": float(outcome.S_exit),
            "T_exit_days": float(outcome.T_exit_days),
            "sigma_exit": float(outcome.sigma_exit),
            "debit_exit": float(outcome.debit_exit),
            "revenue": float(outcome.revenue),
            "fees_exit": float(outcome.fees_exit),
            "pnl_dollars": float(outcome.pnl),
            "win": bool(outcome.win),
            "opt_reason": str(meta.get("opt_reason", "")),
        }
    )

    if opt_structure:
        result["opt_structure"] = str(opt_structure)
    else:
        result["opt_structure"] = spread.structure

    return result

