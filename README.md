# Fraud Detection Project

Fraud Detection Project is a full-stack machine learning application for credit card fraud prediction. It combines a Flask API, a React dashboard, and an ensemble-based training pipeline to score transactions as `Legitimate`, `Suspicious`, or `Fraud`.

## Project Description

Suggested GitHub repository description:

```text
Full-stack credit card fraud detection app with Flask, React, ensemble ML models, Docker support, and GitHub Actions CI.
```

Suggested Docker Hub short description:

```text
Full-stack fraud detection app with a Flask API, React frontend, and ensemble machine learning pipeline.
```

Suggested GitHub topics:

```text
fraud-detection machine-learning flask react docker xgboost lightgbm scikit-learn api fintech
```

## Key Features

- Fraud probability output and human-readable fraud decision
- Risk score from `0` to `100`
- Weighted ensemble of `RandomForest`, `XGBoost`, `LightGBM`, optional `CatBoost`, and `IsolationForest`
- Feature engineering and dynamic thresholding
- Flask API for integration and automation
- React frontend for interactive testing
- Docker image and Compose support
- GitHub Actions workflows for CI and container publishing

## Project Structure

```text
.
+-- app.py
+-- train_pipeline.py
+-- prepare_dataset.py
+-- requirements.txt
+-- Dockerfile
+-- compose.yaml
+-- src/
|   +-- features.py
+-- frontend/
+-- data/
|   +-- cleaned_data.csv
+-- model/
+-- tests/
+-- .github/workflows/ci.yml
+-- .github/workflows/docker-publish.yml
```

## Dataset

The training pipeline expects `data/cleaned_data.csv`.

Required columns:

- `Time`
- `V1` to `V28`
- `Amount`
- `Class`
- `source`

Training uses records where `source == "creditcard"`.

To prepare a compatible dataset from a standard fraud CSV:

```bash
python prepare_dataset.py path/to/creditcard.csv
```

If you already have a compatible file, place it directly at `data/cleaned_data.csv`.

## Local Setup

Install backend dependencies:

```bash
pip install -r requirements.txt
```

Windows fallback:

```powershell
.\Scripts\python.exe -m pip install -r requirements.txt
```

Install frontend dependencies:

```bash
cd frontend
npm ci
```

## Training

Train the ensemble pipeline:

```bash
python train_pipeline.py
```

Enable SMOTE on Windows:

```powershell
$env:USE_SMOTE=1
python train_pipeline.py
```

Optional CatBoost support:

```bash
pip install catboost
```

Artifacts are saved to `model/`.

## Run Locally

Start the Flask app:

```bash
python app.py
```

Windows helpers:

```powershell
.\run.ps1
```

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

Then open `http://localhost:5173`.

## API

Health endpoints:

- `GET /health`
- `GET /api`
- `GET /api/health`

Prediction endpoints:

- `POST /predict`
- `POST /api/predict`

Required JSON fields:

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

Build the image locally:

```bash
docker build -t fraud-detection-app .
```

Run the container:

```bash
docker run --rm -p 5000:5000 fraud-detection-app
```

Or use Docker Compose:

```bash
docker compose up --build
```

App URLs:

- `http://localhost:5000`
- `http://localhost:5000/health`
- `http://localhost:5000/api`

The image build:

- installs Python dependencies
- builds the React frontend
- serves the built frontend through Flask
- exposes port `5000`

## Docker Hub Publishing

This repository now includes [`.github/workflows/docker-publish.yml`](./.github/workflows/docker-publish.yml) for publishing an image to Docker Hub from GitHub Actions.

Before it can push images, add these repository secrets in GitHub:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Then update the `IMAGE_NAME` value in the workflow to match your Docker Hub repository, for example:

```text
your-dockerhub-username/fraud-detection-project
```

After that, pushes to `main` and version tags such as `v1.0.0` can publish ready-to-run images.

## GitHub Actions

The repository includes:

- [`.github/workflows/ci.yml`](./.github/workflows/ci.yml) for backend smoke tests and frontend build validation
- [`.github/workflows/docker-publish.yml`](./.github/workflows/docker-publish.yml) for container image publishing

## Testing

Backend tests:

```bash
python -m unittest discover -s tests -v
```

Frontend production build:

```bash
cd frontend
npm run build
```

## Notes

- `data/cleaned_data.csv` is intentionally not committed
- files in `model/` are generated after training
- prediction requests return `503` if no trained model artifacts are available







https://github.com/user-attachments/assets/4b3c844e-4d11-4b12-a061-05fbcc20a083





