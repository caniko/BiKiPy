

format:
	black .
	isort .
	ruff check bikipy/ --fix
	@echo "Formatting complete 🎉"

mypy:
	poetry run mypy -p bikipy
