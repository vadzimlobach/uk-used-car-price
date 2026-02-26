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