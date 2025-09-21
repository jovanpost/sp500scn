import inspect, importlib, sys, pandas as pd, streamlit as st

st.title("Debug: imports & options columns")

# Show where these modules are actually loading from
import backtest.run_range as rr
import engine.signal_scan as sg
import engine.options_spread as osmod

# Force reload in case Streamlit had an old module object cached
importlib.reload(rr); importlib.reload(sg); importlib.reload(osmod)

st.subheader("Imported file paths")
st.json({
    "run_range_file": inspect.getfile(rr),
    "signal_scan_file": inspect.getfile(sg),
    "options_spread_file": inspect.getfile(osmod),
    "python_exe": sys.executable,
    "sys_path_first": sys.path[:3],  # first few entries only
})

# Sanity: show the 'needed' export columns the RUNNING code expects
try:
    # Re-import to be sure we read from the same module object
    from backtest.run_range import run_range
    src = inspect.getsource(run_range)
    has_options_cols = all(k in src for k in [
        "opt_structure","K1","K2","width_frac","width_pct","T_entry_days","sigma_entry",
        "debit_entry","contracts","cash_outlay","fees_entry","S_exit","T_exit_days",
        "sigma_exit","debit_exit","revenue","fees_exit","pnl_dollars","win",
    ])
    st.subheader("run_range.py contains options columns?")
    st.write("✅ Yes" if has_options_cols else "❌ No (old code running)")
except Exception as e:
    st.error(f"Could not inspect run_range: {e}")

# Bonus: show what OptionsSpreadConfig defaults look like right now
try:
    from engine.options_spread import OptionsSpreadConfig
    st.subheader("OptionsSpreadConfig (defaults)")
    st.json(OptionsSpreadConfig().__dict__)
except Exception as e:
    st.error(f"Could not import OptionsSpreadConfig: {e}")
