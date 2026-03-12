.PHONY: test lint format check train predict precommit-install precommit-run

IMAGE_NAME=car-price
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

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm \
		-v "$(PWD)/artifacts:/app/artifacts" \
		-v "$(PWD)/tests/fixtures:/app/fixtures" \
		$(IMAGE_NAME) \
		--input $(INPUT_PATH)