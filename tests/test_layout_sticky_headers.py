import ui.layout as layout


def test_setup_page_includes_sticky_dataframe_css(monkeypatch):
    calls = []
    html_calls = []
    monkeypatch.setattr(layout.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(layout.st, "markdown", lambda html, *a, **k: calls.append(html))
    monkeypatch.setattr(layout.components.v1, "html", lambda html, *a, **k: html_calls.append(html))
    layout.setup_page()
    css_call = next((c for c in calls if '<style>' in c), '')
    assert 'div[data-testid="stDataFrame"] [role="columnheader"]' in css_call
    assert 'overflow: auto' in css_call
    assert 'position: sticky' in css_call
    assert len(html_calls) == 1


def test_setup_page_includes_table_wrapper_sticky_header_css(monkeypatch):
    calls = []
    monkeypatch.setattr(layout.st, "set_page_config", lambda *a, **k: None)
    monkeypatch.setattr(layout.st, "markdown", lambda html, *a, **k: calls.append(html))
    monkeypatch.setattr(layout.components.v1, "html", lambda *a, **k: None)
    layout.setup_page()
    css_call = next((c for c in calls if '<style>' in c), '')
    assert '.table-wrapper thead th' in css_call
    assert 'position: sticky' in css_call
