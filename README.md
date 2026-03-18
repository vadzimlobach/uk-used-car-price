# UK Used Car Price Predictor

Production-oriented tabular ML pipeline demonstrating reproducible training,
artifact versioning, containerised inference and CI validation.

[![CI](https://github.com/vadzimlobach/uk-used-car-price/actions/workflows/ci.yml/badge.svg)](https://github.com/vadzimlobach/uk-used-car-price/actions/workflows/ci.yml)

## 🧩 Pipeline Overview

This project implements an end-to-end tabular ML pipeline:

1. Raw data preprocessing and validation  
2. Feature engineering and dataset cleaning  
3. Config-driven model training with cross-validation  
4. Artifact versioning for reproducible experiments  
5. Containerised inference runtime  
6. CI pipeline validation (lint → test → Docker smoke test)

Training artifacts are stored under:

```artifacts/runs/<run_id>/```

Each run includes model, metrics, config snapshot and commit hash for traceability.

## 🧱 Tech Stack

- Python 3.11
- scikit-learn
- pandas / NumPy
- Pydantic (input validation)
- Docker (containerised inference)
- GitHub Actions (CI pipeline)
- Ruff + pytest (code quality)

## 🏗 Architecture

Raw CSV → Preprocess → Train (CV) → Artifact store → Docker runtime → CI smoke test 


## 🐳 Containerised Inference

Inference runs inside a lightweight Docker container that loads the latest trained model artifact.

This allows predictions without installing Python dependencies locally.

### 📦 Pull image from GHCR
```bash
docker pull ghcr.io/vadzimlobach/uk-used-car-price:latest
```
Run inference:
```bash
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd):/data" \
  ghcr.io/vadzimlobach/uk-used-car-price:latest \
  --input /data/input.json
```

### 🐳 Run inference via Docker

You can run model inference inside a Docker container without installing Python dependencies locally.

1. Build Docker image
```bash 
docker build -t car-price .
```
2. Train a model (required once)

This step creates a new run directory and updates the pointer file:

```
artifacts/runs/latest_run.txt
```
Example:

```bash
make train
```
3. Run inference

The container automatically resolves the latest trained model if --model is not provided.
```bash
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/tests/fixtures:/app/fixtures" \
  car-price \
  --input /app/fixtures/sample_input.json
```

Optional: override model via environment variable
```bash
docker run --rm \
  -e MODEL_PATH=/app/artifacts/runs/<run_id>/model.joblib \
  -v "$(pwd)/artifacts:/app/artifacts" \
  -v "$(pwd)/tests/fixtures:/app/fixtures" \
  car-price \
  --input /app/fixtures/sample_input.json
```
Example output:

```json
{"predicted_price": 11234.5}
```

## 🚀 Quick Start
### 1️⃣ Setup
```bash 
python -m venv .venv
source .venv/bin/activate
pip install .
```

## ⚙️ CI Pipeline

GitHub Actions validates every push and pull request:

- code formatting and linting
- unit and CLI tests
- Docker image build
- end-to-end inference smoke test using a synthetic dataset

This ensures the container runtime is always functional and reproducible.

### 🏗 Current Project Structure
```bash 
.
├── Dockerfile
├── Makefile
├── README.md
├── artifacts
│   └── runs                            # versioned training runs
│       ├── <run_id>                    
│       │   ├── config.yaml
│       │   ├── cv_summary.json
│       │   ├── figures
│       │   │   ├── residuals_distribution.png
│       │   │   ├── residuals_vs_car_age.png
│       │   │   ├── residuals_vs_mileage_per_year.png
│       │   │   ├── residuals_vs_predicted.png
│       │   │   └── residuals_vs_true.png
│       │   ├── git_commit.txt
│       │   ├── metrics.json
│       │   └── model.joblib
│       └── latest_run.txt              # Latest run pointer
├── configs                             # Training configs
│   └── train.yaml
├── data
│   ├── processed                       # Cleaned dataset
│   └── raw                             # Raw CSV data (multiple brands)
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── scripts
├── src
│   ├── __init__.py
│   ├── analyze.py
│   ├── data_io.py
│   ├── data_loader.py
│   ├── logging_config.py
│   ├── model_utils.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── run_utils.py
│   ├── schema.py                      # 🔒 Single source of truth for inference schema
│   └── train.py
└── tests
    ├── fixtures
    │   └── sample_input.json
    ├── test_data_io.py
    ├── test_data_loader.py
    ├── test_model_utils.py
    ├── test_predict.py
    ├── test_preprocess.py
    ├── test_preprocess_cli.py
    ├── test_run_utils.py
    ├── test_schema.py
    └── test_train.py
```
### 🧹 Data Preprocessing

**The preprocessing pipeline performs:**

 - Target validation

 - Currency cleaning (£, commas)

 - Duplicate row removal

 - Removal of invalid / negative target values

 - Numeric missing value imputation (median)

 - Categorical missing value imputation ("Unknown")

**Run preprocessing:**
``` bash 
python -m src.preprocess \
  --in data/raw/combined.csv \
  --out data/processed/processed.csv \
  --target price
  ```

## 🤖 Model Training

Training is fully config-driven.

Example:

```bash
python -m src.train --config configs/train.yaml
```

*Outputs:*

 - Trained model → *artifacts/runs/<run_id>/model.joblib*

 - Evaluation metrics → *artifacts/runs/<run_id>/metrics.json*

Each training run is fully traceable via stored configuration and git commit hash.
Supports Random Forest and Gradient Boosting variants.

## 📦 Dataset

The original dataset is not included in the repository.

CI uses a small synthetic dataset to validate preprocessing and training steps.

## 🔒 Inference Schema (Single Source of Truth)

*src/schema.py* defines the canonical input contract for predictions.

This ensures:

 - Stable feature ordering

 - Type validation via Pydantic

 - Future API compatibility

 - Prevention of silent inference bugs

All inference paths will validate inputs against this schema.

### 🧪 Testing

The project includes unit and CLI tests covering:

 - Schema validation

 - Target cleaning logic

 - Duplicate handling

 - Missing value imputation

 - Data loading

 - CLI integration

**Run tests:**
```bash
pytest -q
```

## 📊 Example Model Performance

> **Latest Random Forest model:**
>
> RMSE ≈ 1,975
>
> MAE ≈ 1,257
>
> R² ≈ 0.96

(Exact metrics available in artifacts/runs/<run_id>.)

## 🛠 Engineering Practices

This project is structured as an MLOps portfolio piece and includes:

 - Modular src/ architecture

 - Clear separation of code vs artifacts

 - JSON metrics outputs

 - Unit + CLI testing

 - Centralized schema definition

 - Structured logging

## Code Quality

This project uses **Ruff** for linting and formatting, and **pre-commit** for local automated checks.

### Install hooks

```bash
pre-commit install
```

### Common commands
```bash
make lint
make format
make test
make check
make train
make predict
```

## 🚧 Future Improvements

- Publish Docker image to container registry (GHCR)
- Wrap inference as FastAPI service
- Deploy service to AWS / Azure
- Add prediction metadata endpoint
- Implement basic data drift checks
- Add model monitoring hooks

## 🎯 Project Goal

This project demonstrates transition from Test Automation Engineer → MLOps Engineer, focusing on:

 - Reproducibility

 - Code quality

 - Deployment readiness

 - Infrastructure thinking

 - Production mindset for ML systems

### 📌 Next Milestones

 1. FastApi

 2. Cloud deployment (AWS recommended first)

[def]: https://github.com/vadzimlobach/<uk-used-car-price/actions/workflows/ci.yml/badge.svg