import ui.layout as layout

def test_setup_page_includes_sticky_dataframe_css(monkeypatch):
    calls = []
    monkeypatch.setattr(layout.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(layout.st, "markdown", lambda html, *a, **k: calls.append(html))
    layout.setup_page()
    css_call = next((c for c in calls if '<style>' in c), '')
    assert 'div[data-testid="stDataFrame"] thead th' in css_call
    assert 'position: sticky' in css_call
    assert 'tbody td:first-child' in css_call
    assert 'thead th:first-child' in css_call
