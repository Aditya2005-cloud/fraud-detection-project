# Fraud Detection Project

Fraud Detection Project is a full-stack machine learning application for credit card fraud prediction. It includes a Flask API, a React frontend, a training pipeline, and Docker support. The model returns a fraud probability, a risk score from `0` to `100`, and a final decision of `Legitimate`, `Suspicious`, or `Fraud`.

## Short Description

This project helps detect suspicious credit card transactions by combining:

- supervised models
- an unsupervised anomaly detector
- feature engineering
- a weighted ensemble with a dynamic threshold

## Features

- Risk score from `0` to `100`
- Fraud probability output
- Weighted ensemble of `RandomForest`, `XGBoost`, `LightGBM`, optional `CatBoost`, and `IsolationForest`
- Dynamic thresholding for fraud decisions
- React frontend for interactive testing
- Flask API for programmatic use
- Dockerized deployment
- GitHub Actions CI workflow

## Project Structure

```text
.
+-- app.py
+-- train_pipeline.py
+-- prepare_dataset.py
+-- requirements.txt
+-- Dockerfile
+-- src/
|   +-- features.py
+-- frontend/
+-- data/
|   +-- cleaned_data.csv
+-- model/
+-- tests/
+-- .github/workflows/ci.yml
```

## Dataset

The training pipeline expects:

```text
data/cleaned_data.csv
```

The dataset must contain at least:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`
- `source`

The training code uses rows where:

```text
source == "creditcard"
```

### How to get the dataset

Use a credit card fraud CSV with the standard columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`

Then run:

```bash
python prepare_dataset.py path/to/creditcard.csv
```

This creates:

```text
data/cleaned_data.csv
```

If you already have a compatible `cleaned_data.csv`, place it directly inside the `data/` folder.

## Setup

Install backend dependencies:

```bash
pip install -r requirements.txt
```

If Windows uses the wrong interpreter, run:

```powershell
.\Scripts\python.exe -m pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

## Training

Train the ensemble model with:

```bash
python train_pipeline.py
```

Optional SMOTE:

```bash
set USE_SMOTE=1
python train_pipeline.py
```

Optional CatBoost support:

```bash
pip install catboost
```

Training saves artifacts to:

```text
model/
```

## Run Locally

Start the backend:

```bash
python app.py
```

If Windows resolves to global Python, use:

```powershell
.\run.ps1
```

or:

```bat
run.bat
```

Open:

- `http://localhost:5000`
- `http://localhost:5000/health`

For frontend development:

```bash
cd frontend
npm run dev
```

Open:

```text
http://localhost:5173
```

## API

Health endpoints:

- `GET /health`
- `GET /api`
- `GET /api/health`

Prediction endpoints:

- `POST /predict`
- `POST /api/predict`

Required JSON body fields:

- `Time`
- `V1` to `V28`
- `Amount`

Example response:

```json
{
  "risk_score": 17,
  "fraud_probability": 0.17,
  "threshold_used": 0.57,
  "prediction": 0,
  "final_decision": "Legitimate",
  "model": "weighted_ensemble"
}
```

## Docker

Build:

```bash
docker build -t fraud-detection-app .
```

Run:

```bash
docker run --rm -p 5000:5000 fraud-detection-app
```

Open:

```text
http://localhost:5000
```

## Testing

Backend tests:

```bash
python -m unittest discover -s tests -v
```

Frontend build:

```bash
cd frontend
npm run build
```

## GitHub

This repository includes a GitHub Actions workflow at `.github/workflows/ci.yml` to:

- install Python dependencies
- run backend smoke tests
- install frontend dependencies
- build the frontend

## Notes

- `data/cleaned_data.csv` is intentionally not committed because it is large
- trained files in `model/` are generated locally after training
- if model artifacts are missing, prediction requests will fail until training is completed
