.PHONY: test lint format check train predict precommit-install precommit-run

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

check:
	ruff check .
	pytest -q

train:
	python -m src.train --config configs/train.yaml

predict:
	python -m src.predict --help

precommit-install:
	pre-commit install

precommit-run:
	pre-commit run --all-files