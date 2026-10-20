.PHONY: train lint test serve

train:
    python -m src.models.trainer

lint:
    python -m ruff check src/ tests/

test:
    python -m pytest --cov=src tests/

serve:
    python -m src.serving.app