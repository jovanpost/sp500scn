from __future__ import annotations


def test_data_lake_package_exports():
    import data_lake
    from data_lake import Storage, filter_tickers_with_parquet, load_prices_cached
    from data_lake import storage as dl_storage

    assert Storage is dl_storage.Storage
    assert load_prices_cached is dl_storage.load_prices_cached
    assert filter_tickers_with_parquet is dl_storage.filter_tickers_with_parquet
    assert callable(filter_tickers_with_parquet)

    # Package-level __all__ should advertise the helpers for ``from data_lake import *``.
    exported = set(getattr(data_lake, "__all__", ()))
    assert {"Storage", "load_prices_cached", "filter_tickers_with_parquet"}.issubset(exported)
