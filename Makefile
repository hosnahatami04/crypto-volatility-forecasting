.PHONY: install test lint download

install:
	python -m pip install -r requirements.txt

test:
	pytest -v

lint:
	ruff check src tests

download:
	python -m src.data.download
