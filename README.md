# UK Used Car Price Predictor

Production-oriented machine learning project focused on **clean architecture, reproducibility, testing, and MLOps best practices**.

The project trains and evaluates models to predict UK used car prices from structured tabular data and is being incrementally upgraded toward a fully containerized, CI/CD-driven, cloud-deployable service.

[![CI](https://github.com/vadzimlobach/uk-used-car-price/actions/workflows/ci.yml/badge.svg)](https://github.com/vadzimlobach/uk-used-car-price/actions/workflows/ci.yml)

## 🚀 Quick Start
### 1️⃣ Setup
```bash 
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
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
### 🤖 Model Training

**Currently supported models include:**

 - Baseline models

 - Random Forest

 - Gradient Boosting

 - Log-target variants

**Example training run:**
```bash
python -m src.train \
  --in data/processed/processed.csv \
  --model-out artifacts/models/rf.joblib \
  --metrics-out artifacts/reports/rf_metrics.json \
  --model rf \
  --run-name baseline_rf
```

*Outputs:*

 - Trained model → *artifacts/runs/<run_id>/model.joblib*

 - Evaluation metrics → *artifacts/runs/<run_id>/metrics.json*

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

### Planned upgrades:

 - GitHub Actions CI

 - FastAPI service

 - Cloud deployment (AWS / Azure)

## 🎯 Project Goal

This project demonstrates transition from Test Automation Engineer → MLOps Engineer, focusing on:

 - Reproducibility

 - Code quality

 - Deployment readiness

 - Infrastructure thinking

 - Production mindset for ML systems

### 📌 Next Milestones

 1. CI pipeline with lint + tests

 2. FastApi

 3. Cloud deployment (AWS recommended first)

[def]: https://github.com/vadzimlobach/<uk-used-car-price/actions/workflows/ci.yml/badge.svg