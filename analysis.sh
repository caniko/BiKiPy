#!/bin/bash
poetry install
poetry run python -c "from bikipy.ingress.run import analyze_and_save; analyze_and_save('.')"
