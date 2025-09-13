import importlib


def test_backtest_form_renders():
    mod = importlib.import_module('ui.pages.55_Backtest_Range')
    mod.render_page()
