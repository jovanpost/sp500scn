from __future__ import annotations

import json
from typing import Any

import streamlit as st

from engine.precursor_rules import apply_preset_to_session
from engine.scan_shared.precursor_flags import DEFAULT_PARAMS as PRECURSOR_DEFAULTS


def _clamp(value: float, *, minimum: float, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(minimum)
    if maximum is not None:
        result = min(maximum, result)
    return max(minimum, result)


def render_precursor_section(session: Any) -> None:
    """Render the shared Spike Precursor controls used by lab + scanner pages."""

    default_enabled = bool(session.get("scanner_precursor_enabled", False))
    st.checkbox(
        "Enable Spike Precursor filters",
        key="scanner_precursor_enabled",
        value=default_enabled,
        help="When enabled, candidate trades must match at least one precursor flag.",
    )

    with st.expander("Spike Precursor Filters (optional)", expanded=False):
        precursors_enabled = bool(session.get("scanner_precursor_enabled", False))
        disabled_children = not precursors_enabled

        preset_upload = st.file_uploader(
            "Import from Spike Lab preset",
            type=["json"],
            key="scanner_precursor_preset",
            help="Apply a preset exported from the Spike Precursor Lab.",
        )
        if preset_upload is not None:
            try:
                preset_raw = preset_upload.read()
                preset_data = json.loads(preset_raw.decode("utf-8")) if preset_raw else {}
                applied = apply_preset_to_session(preset_data, session)
            except Exception:
                st.error("Could not read preset JSON. Please check the file format.")
            else:
                if applied:
                    st.success(
                        "Preset applied: " + ", ".join(sorted({flag for flag in applied}))
                    )
                else:
                    st.warning("Preset contained no supported precursor flags.")

        slider_default = int(
            max(
                1,
                min(
                    60,
                    float(
                        session.get(
                            "scanner_precursor_within", PRECURSOR_DEFAULTS["lookback_days"]
                        )
                        or 1
                    ),
                ),
            )
        )
        st.slider(
            "Look back within N business days",
            min_value=1,
            max_value=60,
            value=slider_default,
            key="scanner_precursor_within",
            disabled=disabled_children,
        )

        logic_options = ("ANY", "ALL")
        logic_default = str(session.get("scanner_precursor_logic", "ANY") or "ANY").upper()
        if logic_default not in logic_options:
            logic_default = "ANY"
        st.radio(
            "Logic mode",
            options=logic_options,
            index=logic_options.index(logic_default),
            key="scanner_precursor_logic",
            disabled=disabled_children,
            horizontal=True,
        )

        st.markdown("**Trend & Momentum**")
        trend_cols = st.columns(3)
        trend_cols[0].checkbox(
            "EMA 20/50 cross up",
            key="scanner_precursor_ema",
            disabled=disabled_children,
        )
        trend_cols[1].checkbox(
            "RSI cross ≥ 50",
            key="scanner_precursor_rsi50",
            disabled=disabled_children,
        )
        trend_cols[2].checkbox(
            "RSI cross ≥ 60",
            key="scanner_precursor_rsi60",
            disabled=disabled_children,
        )

        st.markdown("**Volatility squeezes**")
        squeeze_cols = st.columns(2)
        with squeeze_cols[0]:
            st.checkbox(
                "ATR percentile ≤",
                key="scanner_precursor_atr",
                disabled=disabled_children,
            )
            atr_value = _clamp(
                session.get(
                    "scanner_precursor_atr_threshold", PRECURSOR_DEFAULTS["atr_pct_threshold"]
                ),
                minimum=1.0,
                maximum=100.0,
            )
            st.number_input(
                "ATR percentile",
                min_value=1.0,
                max_value=100.0,
                step=1.0,
                value=float(atr_value),
                key="scanner_precursor_atr_threshold",
                disabled=disabled_children or not session.get("scanner_precursor_atr", False),
            )
        with squeeze_cols[1]:
            st.checkbox(
                "BB width percentile ≤",
                key="scanner_precursor_bb",
                disabled=disabled_children,
            )
            bb_value = _clamp(
                session.get(
                    "scanner_precursor_bb_threshold", PRECURSOR_DEFAULTS["bb_pct_threshold"]
                ),
                minimum=1.0,
                maximum=100.0,
            )
            st.number_input(
                "BB percentile",
                min_value=1.0,
                max_value=100.0,
                step=1.0,
                value=float(bb_value),
                key="scanner_precursor_bb_threshold",
                disabled=disabled_children or not session.get("scanner_precursor_bb", False),
            )

        st.markdown("**Range & breakouts**")
        range_cols = st.columns(3)
        range_cols[0].checkbox(
            "NR7",
            key="scanner_precursor_nr7",
            disabled=disabled_children,
        )
        range_cols[1].checkbox(
            "New high 20",
            key="scanner_precursor_high20",
            disabled=disabled_children,
        )
        range_cols[2].checkbox(
            "New high 63",
            key="scanner_precursor_high63",
            disabled=disabled_children,
        )

        st.checkbox(
            "Support/resistance ratio ≥ 2",
            key="scanner_precursor_sr",
            disabled=disabled_children,
        )

        st.markdown("**Gaps & volume**")
        gv_cols = st.columns(2)
        with gv_cols[0]:
            st.checkbox(
                "Prior-day gap ≥ %",
                key="scanner_precursor_gap",
                disabled=disabled_children,
            )
            gap_value = _clamp(
                session.get(
                    "scanner_precursor_gap_threshold", PRECURSOR_DEFAULTS["gap_min_pct"]
                ),
                minimum=0.0,
            )
            st.number_input(
                "Gap percent",
                min_value=0.0,
                max_value=50.0,
                step=0.5,
                value=float(gap_value),
                key="scanner_precursor_gap_threshold",
                disabled=disabled_children or not session.get("scanner_precursor_gap", False),
            )
        with gv_cols[1]:
            vol_value = _clamp(
                session.get(
                    "scanner_precursor_vol_threshold", PRECURSOR_DEFAULTS["vol_min_mult"]
                ),
                minimum=0.1,
            )
            st.number_input(
                "Volume multiple",
                min_value=0.1,
                max_value=20.0,
                step=0.1,
                value=float(vol_value),
                key="scanner_precursor_vol_threshold",
                disabled=disabled_children,
            )
            st.checkbox(
                "Day -1 volume ≥ threshold",
                key="scanner_precursor_vol_d1",
                disabled=disabled_children,
            )
            st.checkbox(
                "Day -2 volume ≥ threshold",
                key="scanner_precursor_vol_d2",
                disabled=disabled_children,
            )


__all__ = ["render_precursor_section"]
