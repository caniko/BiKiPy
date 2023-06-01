

format:
	black .
	isort .
	ruff check . --fix

mypy:
	poetry run mypy -p bikipy
