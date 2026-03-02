# UK Used Car Price Predictor

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🏗 Project Structure
```
uk-used-car-price/
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── logging_config.py
├── tests/
├── pytest.ini
├── requirements.txt

```

## 🧹 Preprocessing Pipeline

> - The preprocessing step performs:
> - Target validation
> - Currency cleaning (£, commas)
> - Duplicate row removal
> - Removal of invalid / negative target values
> - Numeric missing value imputation (median)
> - Categorical missing value imputation ("Unknown")
>
> ### Run:
>```bash
>python -m src.preprocess \
>  --in data/raw/input.csv \
>  --out data/processed/processed.csv \
>  --target price
>```
> ```bash
> python3 -m src.train --in data/processed/processed.csv --model-out src/models/rf.joblib --metrics-out src/reports/rf_metrics.json --model rf --run-name baseline_rf
>```
>```bash
> python3 -m src.preprocess --in data/raw/combined.csv --out data/processed/processed.csv
>```
>```bash
> python3 -m src.data_loader --path data/raw
>```

## 🧪Testing
>Unit tests cover:
> - Schema validation. 
> - Target cleaning logic. 
> - Duplicate handling. 
> - Missing value imputation. 
> - CLI integration. 
>
> ### Run tests:
>```bash
>    python -m pytest -q
>```