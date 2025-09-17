.PHONY: test smoke-adi smoke-one

LAKE_ROOT ?=

test:
	pytest -q

smoke-adi:
	python scripts/smoke_raw_ohlc.py --ticker ADI --date 2020-03-20 $(if $(LAKE_ROOT),--lake-root $(LAKE_ROOT),)

# usage: make smoke-one TKR=MSFT DAY=2020-12-10 [LAKE_ROOT=/path/to/lake]
smoke-one:
	python scripts/smoke_raw_ohlc.py --ticker $(TKR) --date $(DAY) $(if $(LAKE_ROOT),--lake-root $(LAKE_ROOT),)
