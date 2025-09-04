import ui.layout as layout

def test_setup_page_includes_sticky_dataframe_css(monkeypatch):
    calls = []
    monkeypatch.setattr(layout.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(layout.st, "markdown", lambda html, *a, **k: calls.append(html))
    layout.setup_page()
    css_call = next((c for c in calls if '<style>' in c), '')
    assert 'div[data-testid="stDataFrame"] {' in css_call
    assert 'position: relative' in css_call
    assert 'overflow: auto' in css_call
    assert 'div[data-testid="stDataFrame"] thead th' in css_call
    assert 'div[data-testid="stDataFrame"] [role="columnheader"]' in css_call
    assert 'position: sticky' in css_call


def test_setup_page_includes_table_wrapper_sticky_header_css(monkeypatch):
    calls = []
    monkeypatch.setattr(layout.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(layout.st, "markdown", lambda html, *a, **k: calls.append(html))
    layout.setup_page()
    css_call = next((c for c in calls if '<style>' in c), '')
    assert '.table-wrapper thead th' in css_call
    assert 'position: sticky' in css_call


def test_setup_page_injects_sticky_helper(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(layout.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(layout.st, "markdown", lambda html, *a, **k: calls.append(html))
    layout.setup_page()
    js_calls = [c for c in calls if "__stickyAudit__" in c]
    assert len(js_calls) == 1
