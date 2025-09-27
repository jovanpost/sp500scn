from __future__ import annotations

import pytest

from engine.precursor_rules import apply_preset_to_session, build_conditions_from_session


class SessionDict(dict):
    """Simple session_state stand-in supporting attribute access."""

    def __getattr__(self, item: str):  # pragma: no cover - support streamlit style
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    def get(self, key, default=None):  # type: ignore[override]
        return super().get(key, default)


def test_apply_preset_sets_expected_keys():
    session = SessionDict()
    preset = {
        "logic": "all",
        "within_days": 7,
        "conditions": [
            {"flag": "atr_squeeze_q", "max_percentile": 15},
            {"flag": "vol_d1", "min_mult": 2.0},
        ],
    }

    applied = apply_preset_to_session(preset, session)

    assert session["scanner_precursor_enabled"] is True
    assert session["scanner_precursor_logic"] == "ALL"
    assert session["scanner_precursor_within"] == 7
    assert "atr_squeeze_pct" in applied
    assert "vol_mult_d1_ge_x" in applied
    assert session["scanner_precursor_atr_threshold"] == pytest.approx(15)
    assert session["scanner_precursor_vol_threshold"] == pytest.approx(2.0)


def test_build_conditions_from_session_round_trip():
    session = SessionDict(
        scanner_precursor_enabled=True,
        scanner_precursor_within=10,
        scanner_precursor_logic="ANY",
        scanner_precursor_atr=True,
        scanner_precursor_atr_threshold=20,
        scanner_precursor_gap=True,
        scanner_precursor_gap_threshold=4,
        scanner_precursor_vol_d2=True,
        scanner_precursor_vol_threshold=1.5,
    )
    payload = build_conditions_from_session(session)
    assert payload is not None
    assert payload["within_days"] == 10
    assert payload["logic"] == "ANY"
    flags = {cond["flag"] for cond in payload["conditions"]}
    assert {"atr_squeeze_pct", "gap_up_ge_gpct_prev", "vol_mult_d2_ge_x"} <= flags
    assert payload["atr_pct_threshold"] == pytest.approx(20.0)
    assert payload["gap_min_pct"] == pytest.approx(4.0)
    assert payload["vol_min_mult"] == pytest.approx(1.5)


def test_build_conditions_returns_none_when_disabled():
    session = SessionDict(scanner_precursor_enabled=False)
    assert build_conditions_from_session(session) is None
