from contextlib import contextmanager

import pandas as pd

from ui.components import tables


def test_show_df_callable(monkeypatch):
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(tables.st, "subheader", lambda *a, **k: calls.append(("subheader", a, k)))
    monkeypatch.setattr(tables.st, "info", lambda *a, **k: calls.append(("info", a, k)))
    monkeypatch.setattr(tables.st, "dataframe", lambda *a, **k: calls.append(("dataframe", a, k)))
    monkeypatch.setattr(tables.st, "download_button", lambda *a, **k: calls.append(("download_button", a, k)))
    monkeypatch.setattr(tables.st, "code", lambda *a, **k: calls.append(("code", a, k)))

    @contextmanager
    def dummy_expander(*args, **kwargs):
        calls.append(("expander", args, kwargs))
        yield

    monkeypatch.setattr(tables.st, "expander", dummy_expander)

    df = pd.DataFrame({"a": [1, 2]})

    tables.show_df("Example", df, "example")

    assert any(name == "dataframe" for name, *_ in calls)
    assert any(name == "download_button" for name, *_ in calls)
    assert any(name == "code" for name, *_ in calls)
