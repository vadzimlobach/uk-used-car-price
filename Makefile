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

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm \
		-v "$(PWD)/artifacts:/app/artifacts" \
		-v "$(PWD)/tests/fixtures:/app/fixtures" \
		$(IMAGE_NAME) \
		--input $(INPUT_PATH)

docker-run-env:
	docker run --rm \
		-e MODEL_PATH=/app/artifacts/runs/20260305_115316_rf_baseline/model.joblib \
		-v "$(PWD)/artifacts:/app/artifacts" \
		-v "$(PWD)/tests/fixtures:/app/fixtures" \
		$(IMAGE_NAME) \
		--input $(INPUT_PATH)

docker-run-api:
	docker run -p 8000:8000 \
  		-v "$(PWD)/artifacts:/app/artifacts" \
  		car-price-api