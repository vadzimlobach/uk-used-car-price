FROM python:3.11-slim

ARG GIT_COMMIT=unknown

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GIT_COMMIT=$GIT_COMMIT \
    MODEL_PATH=/app/artifacts/serving/model.joblib \
    MODEL_METADATA_PATH=/app/artifacts/serving/metadata.json

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src/ src/
COPY artifacts/serving artifacts/serving
COPY configs/ configs/

RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]