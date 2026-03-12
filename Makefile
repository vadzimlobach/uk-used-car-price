.PHONY: test lint format check train predict precommit-install precommit-run

IMAGE_NAME=car-price
LATEST_RUN=$(shell cat artifacts/runs/latest_run.txt)
MODEL_PATH=/app/artifacts/runs/$(LATEST_RUN)/model.joblib
INPUT_PATH=/app/fixtures/sample_input.json

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

check-latest-run:
	@test -f artifacts/runs/latest_run.txt || (echo "latest_run.txt not found. Train model first."; exit 1)

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run: check-latest-run
	docker run --rm \
		-v "$(PWD)/artifacts:/app/artifacts" \
		-v "$(PWD)/tests/fixtures:/app/fixtures" \
		$(IMAGE_NAME) \
		--model $(MODEL_PATH) \
		--input $(INPUT_PATH)