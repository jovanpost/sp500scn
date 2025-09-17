.PHONY: test smoke-adi smoke-one migrate-raw verify-one

LAKE_ROOT ?=

test:
	pytest -q

smoke-adi:
	python scripts/smoke_raw_ohlc.py --ticker ADI --date 2020-03-20 $(if $(LAKE_ROOT),--lake-root $(LAKE_ROOT),)

# usage: make smoke-one TKR=MSFT DAY=2020-12-10 [LAKE_ROOT=/path/to/lake]
smoke-one:
	python scripts/smoke_raw_ohlc.py --ticker $(TKR) --date $(DAY) $(if $(LAKE_ROOT),--lake-root $(LAKE_ROOT),)

migrate-raw:
	python scripts/migrate_lake_to_raw.py --tickers $(TKRS) $(if $(START),--start $(START),) $(if $(END),--end $(END),)

# Usage: make verify-one TKR=ALB START=2025-02-01 END=2025-04-01
verify-one:
	python scripts/verify_raw_vs_yahoo.py --ticker $(TKR) $(if $(START),--start $(START),) $(if $(END),--end $(END),)
