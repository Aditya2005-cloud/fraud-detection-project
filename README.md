# Fraud Detection Project

API and training pipeline for fraud detection with **risk score (0–100)**, **dynamic threshold**, and **weighted ensemble** (supervised + Isolation Forest).

## Features

- **Risk score 0–100** instead of only 0/1 prediction  
- **Dynamic threshold** tuned for F2 (favors recall on fraud)  
- **Weighted ensemble**: RandomForest, XGBoost, LightGBM, CatBoost (supervised) + Isolation Forest (unsupervised)  
- **Feature engineering**: amount buckets, time-based (hour, sin/cos), transaction velocity  
- **Cost-sensitive learning**: `class_weight="balanced"`, `scale_pos_weight`; optional SMOTE  
- **Cross-validation** and **hyperparameter tuning** (RandomizedSearchCV; GridSearchCV optional)  

## Setup

```bash
pip install -r requirements.txt
```

Optional: install CatBoost for the full ensemble (`pip install catboost`).

## Training (new pipeline)

From the project root:

```bash
python train_pipeline.py
```

This will:

1. Load credit-card data from `data/cleaned_data.csv`  
2. Engineer features (amount buckets, time, velocity)  
3. Run 5-fold CV and RandomizedSearchCV on RandomForest  
4. Train RF, XGBoost, LightGBM, CatBoost, Isolation Forest  
5. Tune the decision threshold (F2) on the ensemble score  
6. Save artifacts to `model/` (including `ensemble_artifacts.pkl`)

Optional SMOTE (oversample fraud class):

```bash
set USE_SMOTE=1
python train_pipeline.py
```

## React frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The app proxies API requests to the backend (port 5000). Use "Load legitimate sample" or "Load fraud sample", then "Check risk" to see risk score (0–100) and decision.

Build for production: `npm run build` (output in `frontend/dist`).

## API

Start the backend:

```bash
python app.py
```

- **GET /** – health check  
- **POST /predict** – JSON body with keys: `Time`, `V1`–`V28`, `Amount`  

Response includes:

- `risk_score` (0–100)  
- `fraud_probability`  
- `threshold_used` (dynamic)  
- `prediction` (0/1)  
- `final_decision`: "Fraud" | "Suspicious" | "Legitimate"  
- `model`: `"weighted_ensemble"` or `"legacy"`  

If `model/ensemble_artifacts.pkl` is missing, the app falls back to `fraud_model.pkl` and optional `isolation_forest.pkl` (legacy).

## Project layout

- `app.py` – Flask API (ensemble or legacy)  
- `frontend/` – React (Vite + TypeScript) UI for predictions  
- `train_pipeline.py` – full training with CV, tuning, ensemble, threshold  
- `src/features.py` – feature engineering (shared by train and app)  
- `data/cleaned_data.csv` – cleaned data (credit card + other sources)  
- `model/` – saved models and `ensemble_artifacts.pkl`  
- `notebooks/model_training.ipynb` – original exploratory training  
