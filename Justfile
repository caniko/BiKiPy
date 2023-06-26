

format:
	black .
	isort .
	ruff check bikipy/ --fix
	@echo "Formatting complete 🎉"

mypy:
	mypy -p bikipy
