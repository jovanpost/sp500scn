import ui.layout as layout

def test_setup_page_includes_sticky_dataframe_css(monkeypatch):
    calls = []
    monkeypatch.setattr(layout.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(layout.st, "markdown", lambda html, *a, **k: calls.append(html))
    layout.setup_page()
    css_call = next((c for c in calls if '<style>' in c), '')
    assert 'div[data-testid="stDataFrame"] th[role="columnheader"]' in css_call
    assert 'position: sticky' in css_call
    assert 'th[role="rowheader"]' in css_call
