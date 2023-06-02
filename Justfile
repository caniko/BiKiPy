

format:
	black .
	isort .
	ruff check . --fix
	@echo "Formatting complete 🎉"

mypy:
	mypy -p bikipy
