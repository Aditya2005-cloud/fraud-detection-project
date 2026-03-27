"""
Fraud Detection API.
Uses weighted ensemble (RF, XGBoost, LightGBM, CatBoost, Isolation Forest) when available,
with risk score 0-100 and dynamic threshold. Falls back to legacy models if ensemble not trained.
"""
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIST_DIR = PROJECT_ROOT / "frontend" / "dist"
MODEL_DIR = PROJECT_ROOT / "model"
ARTIFACTS_PATH = MODEL_DIR / "ensemble_artifacts.pkl"

app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST_DIR) if FRONTEND_DIST_DIR.exists() else None,
    static_url_path="",
)
CORS(app)

# Base feature order for API input (before feature engineering)
FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount",
]

# -----------------------------
# Load models: prefer ensemble, else legacy
# -----------------------------
ensemble_artifacts = None
legacy_supervised = None
legacy_iso = None

try:
    ensemble_artifacts = joblib.load(ARTIFACTS_PATH)
except Exception:
    pass

if ensemble_artifacts is None:
    try:
        legacy_supervised = joblib.load(MODEL_DIR / "fraud_model.pkl")
    except Exception:
        legacy_supervised = None
    try:
        legacy_iso = joblib.load(MODEL_DIR / "isolation_forest.pkl")
    except Exception:
        legacy_iso = None


def _force_single_thread(model):
    """Disable estimator-level parallelism for safer inference across environments."""
    if model is None:
        return None

    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
        except Exception:
            pass

    if hasattr(model, "set_params"):
        try:
            params = model.get_params(deep=False)
        except Exception:
            params = {}
        updates = {}
        if "n_jobs" in params:
            updates["n_jobs"] = 1
        if "thread_count" in params:
            updates["thread_count"] = 1
        if "nthread" in params:
            updates["nthread"] = 1
        if updates:
            try:
                model.set_params(**updates)
            except Exception:
                pass

    return model


def _prepare_runtime_models():
    """Normalize loaded estimators so local and container inference are consistent."""
    global ensemble_artifacts, legacy_supervised, legacy_iso

    if ensemble_artifacts is not None:
        for name in ("rf", "xgb", "lgb", "iso", "catboost"):
            if name in ensemble_artifacts:
                ensemble_artifacts[name] = _force_single_thread(ensemble_artifacts[name])

    legacy_supervised = _force_single_thread(legacy_supervised)
    legacy_iso = _force_single_thread(legacy_iso)


_prepare_runtime_models()


def _iso_score_to_prob(scores):
    """Map Isolation Forest decision function (negative = anomaly) to [0,1]."""
    min_s, max_s = scores.min(), scores.max()
    if max_s <= min_s:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def _ensemble_predict_proba(X_s, artifacts):
    """Weighted ensemble probability from all models."""
    w = artifacts["weights"]
    feature_frame = pd.DataFrame(X_s, columns=artifacts["feature_order"])
    p = w["rf"] * artifacts["rf"].predict_proba(X_s)[:, 1]
    p += w["xgb"] * artifacts["xgb"].predict_proba(X_s)[:, 1]
    p += w["lgb"] * artifacts["lgb"].predict_proba(feature_frame)[:, 1]
    p += w["iso"] * _iso_score_to_prob(-artifacts["iso"].decision_function(X_s))
    if artifacts.get("catboost") is not None:
        p += w["catboost"] * artifacts["catboost"].predict_proba(X_s)[:, 1]
    return p


def _probability_to_risk_score(prob, scale=100):
    """Convert probability to risk score 0-100."""
    return int(round(np.clip(float(prob), 0, 1) * scale))


@app.route("/", methods=["GET"])
def home():
    if FRONTEND_DIST_DIR.exists():
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"status": "ok", "message": "Fraud Detection API is running"})


@app.route("/health", methods=["GET"])
@app.route("/api", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health():
    model_mode = "weighted_ensemble" if ensemble_artifacts is not None else "legacy"
    return jsonify(
        {
            "status": "ok",
            "model_mode": model_mode,
            "frontend_built": FRONTEND_DIST_DIR.exists(),
        }
    )


@app.route("/predict", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        return jsonify({"error": "Missing features", "missing": missing}), 400

    # Single row as array (base features only)
    base_features = np.array([[data[f] for f in FEATURE_ORDER]], dtype=float)

    if ensemble_artifacts is not None:
        # Ensemble path: feature engineering + weighted ensemble + risk score + dynamic threshold.
        fe = ensemble_artifacts["feature_engineer"]
        scaler = ensemble_artifacts["scaler"]
        feature_order = ensemble_artifacts["feature_order"]
        threshold = ensemble_artifacts["threshold"]

        # Add engineered features (velocity uses fitted median at inference).
        X_df = fe.transform(base_features, velocity_override=None)
        X = X_df[feature_order].values
        X_s = scaler.transform(X)

        prob = _ensemble_predict_proba(X_s, ensemble_artifacts)[0]
        risk_score = _probability_to_risk_score(prob)
        binary_pred = 1 if prob >= threshold else 0

        if binary_pred == 1:
            final_decision = "Fraud"
        elif risk_score >= 50:
            final_decision = "Suspicious"
        else:
            final_decision = "Legitimate"

        return jsonify(
            {
                "risk_score": risk_score,
                "fraud_probability": float(prob),
                "threshold_used": float(threshold),
                "prediction": binary_pred,
                "final_decision": final_decision,
                "model": "weighted_ensemble",
            }
        )

    # Legacy path: single supervised model + optional Isolation Forest.
    if legacy_supervised is None:
        return jsonify({"error": "No trained model artifacts were loaded"}), 503

    probs = legacy_supervised.predict_proba(base_features)[:, 1]
    threshold = 0.3
    supervised_pred = int(probs[0] >= threshold)
    risk_score = _probability_to_risk_score(probs[0])

    response = {
        "risk_score": risk_score,
        "fraud_probability": float(probs[0]),
        "threshold_used": float(threshold),
        "supervised_prediction": supervised_pred,
        "model": "legacy",
    }

    if legacy_iso is not None:
        iso_pred = legacy_iso.predict(base_features)
        iso_flag = 1 if iso_pred[0] == -1 else 0
        response["unsupervised_prediction"] = iso_flag
        if supervised_pred == 1:
            final_decision = "Fraud"
        elif iso_flag == 1:
            final_decision = "Suspicious"
        else:
            final_decision = "Legitimate"
    else:
        final_decision = "Fraud" if supervised_pred == 1 else "Legitimate"

    response["prediction"] = supervised_pred
    response["final_decision"] = final_decision
    return jsonify(response)


@app.route("/<path:path>", methods=["GET"])
def frontend(path):
    if FRONTEND_DIST_DIR.exists():
        asset_path = FRONTEND_DIST_DIR / path
        if asset_path.exists() and asset_path.is_file():
            return send_from_directory(app.static_folder, path)
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
