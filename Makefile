.PHONY: test lint format check train predict precommit-install precommit-run

IMAGE_NAME=car-price-api
INPUT_PATH=/app/fixtures/sample_input.json

test:
	pytest -q

lint:
	ruff check .

fix:
	ruff check . --fix

format:
	ruff format .

format-check:
	ruff format --check .

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

prepare-serving-artifacts:
	python scripts/prepare_serving_artifacts.py

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm \
		-v "$(PWD)/tests/fixtures:/app/fixtures" \
		$(IMAGE_NAME) \
		--input /app/fixtures/sample_input.json

docker-run-api:
	docker run --rm -p 8000:8000 \
		car-price-api

docker-run-dev:
	docker run --rm \
		-e MODEL_PATH=/app/artifacts/runs/$(RUN_ID)/model.joblib \
		-v "$(PWD)/artifacts:/app/artifacts" \
		-v "$(PWD)/tests/fixtures:/app/fixtures" \
		$(IMAGE_NAME) \
		--input /app/fixtures/sample_input.json