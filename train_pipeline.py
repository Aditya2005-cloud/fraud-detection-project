"""
Fraud detection training pipeline:
- Feature engineering (amount buckets, time-based, transaction velocity)
- Cross-validation evaluation
- Hyperparameter tuning (GridSearch / RandomizedSearch)
- Cost-sensitive learning for class imbalance
- Models: RandomForest, XGBoost, LightGBM, CatBoost (supervised) + Isolation Forest (unsupervised)
- Weighted ensemble + dynamic threshold tuning
- Risk score 0-100 and saved artifacts for the API
"""
import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    roc_auc_score,
)

# Optional: imblearn for SMOTE / cost-sensitive
try:
    from imblearn.over_sampling import SMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

import xgboost as xgb
import lightgbm as lgb

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.features import FraudFeatureEngineer, get_feature_order, BASE_FEATURE_ORDER

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_data.csv"
MODEL_DIR = PROJECT_ROOT / "model"
RANDOM_STATE = 42
N_JOBS = -1

# ---------- Data loading ----------
def load_creditcard_data(path=DATA_PATH):
    df = pd.read_csv(path, low_memory=False)
    df = df[df["source"] == "creditcard"].copy()
    cols = BASE_FEATURE_ORDER + ["Class"]
    df = df[cols]
    df["Class"] = df["Class"].astype(int)
    return df


# ---------- Feature pipeline ----------
def build_feature_pipeline(X_raw, fe):
    X_df = fe.transform(X_raw)
    cols = get_feature_order()
    return X_df[cols].values, cols


# ---------- Cross-validation evaluation ----------
def evaluate_cv(model, X, y, cv=5, scoring="f2"):
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    if scoring == "f2":
        scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    else:
        scorer = scoring
    return cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=N_JOBS)


# ---------- Dynamic threshold tuning ----------
def tune_threshold(y_true, y_proba, metric="f2", thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.95, 0.02)
    best_t, best_score = 0.2, 0.0
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return best_t, best_score


# ---------- Risk score 0-100 ----------
def probability_to_risk_score(prob, scale=100):
    return int(round(np.clip(prob, 0, 1) * scale))


# ---------- Isolation Forest: anomaly score to 0-1 (higher = more anomalous) ----------
def iso_score_to_prob(scores):
    # scores are negative (more negative = more anomalous); map to [0,1]
    min_s, max_s = scores.min(), scores.max()
    if max_s <= min_s:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading data...")
    df = load_creditcard_data()
    X_raw = df.drop("Class", axis=1)
    y = df["Class"].values

    print("Feature engineering...")
    fe = FraudFeatureEngineer(amount_quantiles=(0.33, 0.66), velocity_window_sec=3600)
    fe.fit(X_raw)
    X, feature_names = build_feature_pipeline(X_raw, fe)
    feature_list = list(feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Optional: SMOTE for cost-sensitive learning (oversample minority)
    if HAS_IMBLEARN and os.environ.get("USE_SMOTE", "").lower() in ("1", "true", "yes"):
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Applied SMOTE oversampling.")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Class weight for imbalance (cost-sensitive)
    n_neg, n_pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    # ---------- Cross-validation evaluation (single model example) ----------
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for train_idx, val_idx in skf.split(X_train_s, y_train):
        X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        m = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS)
        m.fit(X_tr, y_tr)
        p = m.predict_proba(X_val)[:, 1]
        cv_scores.append(roc_auc_score(y_val, p))
    print(f"RF 5-fold CV ROC-AUC: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

    # ---------- Train supervised models (with cost-sensitive / class_weight) ----------
    print("Training RandomForest (class_weight=balanced)...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    rf.fit(X_train_s, y_train)

    print("Training XGBoost (scale_pos_weight)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    )
    xgb_model.fit(X_train_s, y_train)

    print("Training LightGBM (scale_pos_weight)...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        verbose=-1,
        n_jobs=N_JOBS,
    )
    lgb_model.fit(X_train_s, y_train, feature_name=feature_list)

    if HAS_CATBOOST:
        print("Training CatBoost (scale_pos_weight)...")
        cb_model = cb.CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.05,
            scale_pos_weight=scale_pos,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        cb_model.fit(X_train_s, y_train)
    else:
        cb_model = None
        print("CatBoost not installed; skipping.")

    # ---------- Isolation Forest (unsupervised) ----------
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=min(0.01, y_train.mean() * 2),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    iso.fit(X_train_s)

    # ---------- Ensemble: weighted average of probabilities ----------
    # Weights: can be tuned; here use equal for supervised and one for iso
    weights = {
        "rf": 0.25,
        "xgb": 0.25,
        "lgb": 0.25,
        "catboost": 0.15 if HAS_CATBOOST else 0.0,
        "iso": 0.10,
    }
    if not HAS_CATBOOST:
        weights["rf"] += 0.05
        weights["xgb"] += 0.05

    def ensemble_predict_proba(X_s):
        p_rf = rf.predict_proba(X_s)[:, 1]
        p_xgb = xgb_model.predict_proba(X_s)[:, 1]
        p_lgb = lgb_model.predict_proba(X_s)[:, 1]
        p_iso = iso_score_to_prob(-iso.decision_function(X_s))  # higher = more anomaly
        out = weights["rf"] * p_rf + weights["xgb"] * p_xgb + weights["lgb"] * p_lgb + weights["iso"] * p_iso
        if HAS_CATBOOST:
            p_cb = cb_model.predict_proba(X_s)[:, 1]
            out += weights["catboost"] * p_cb
        return out

    # ---------- Dynamic threshold on ensemble probability ----------
    train_ens_proba = ensemble_predict_proba(X_train_s)
    opt_threshold, _ = tune_threshold(y_train, train_ens_proba, metric="f2")
    print(f"Optimal threshold (F2): {opt_threshold:.3f}")

    # ---------- Evaluation ----------
    test_ens_proba = ensemble_predict_proba(X_test_s)
    y_pred_binary = (test_ens_proba >= opt_threshold).astype(int)
    print("\nEnsemble (dynamic threshold) on test set:")
    print(confusion_matrix(y_test, y_pred_binary))
    print(classification_report(y_test, y_pred_binary))
    print(f"ROC-AUC: {roc_auc_score(y_test, test_ens_proba):.4f}")

    # ---------- Hyperparameter tuning (RandomizedSearchCV; optional GridSearchCV) ----------
    print("\nRunning RandomizedSearchCV on RandomForest (small grid)...")
    param_dist = {
        "n_estimators": [150, 200, 250],
        "max_depth": [8, 10, 12],
        "min_samples_leaf": [3, 5, 8],
    }
    # For GridSearchCV use: GridSearchCV(..., param_grid=param_grid, cv=3, ...)
    rf_tuned = RandomizedSearchCV(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS),
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring=make_scorer(fbeta_score, beta=2, zero_division=0),
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    rf_tuned.fit(X_train_s, y_train)
    print(f"Best RF params: {rf_tuned.best_params_}")
    # Optionally replace rf with rf_tuned.best_estimator_ for production
    rf_final = rf_tuned.best_estimator_

    # Rebuild ensemble with tuned RF and recompute threshold (optional; keep simple here)
    def ensemble_predict_proba_final(X_s):
        p_rf = rf_final.predict_proba(X_s)[:, 1]
        p_xgb = xgb_model.predict_proba(X_s)[:, 1]
        p_lgb = lgb_model.predict_proba(X_s)[:, 1]
        p_iso = iso_score_to_prob(-iso.decision_function(X_s))
        out = weights["rf"] * p_rf + weights["xgb"] * p_xgb + weights["lgb"] * p_lgb + weights["iso"] * p_iso
        if HAS_CATBOOST:
            out += weights["catboost"] * cb_model.predict_proba(X_s)[:, 1]
        return out

    train_ens_proba_final = ensemble_predict_proba_final(X_train_s)
    opt_threshold_final, _ = tune_threshold(y_train, train_ens_proba_final, metric="f2")
    test_ens_proba_final = ensemble_predict_proba_final(X_test_s)
    y_pred_final = (test_ens_proba_final >= opt_threshold_final).astype(int)
    print("\nEnsemble with tuned RF + dynamic threshold:")
    print(confusion_matrix(y_test, y_pred_final))
    print(classification_report(y_test, y_pred_final))

    # ---------- Save artifacts for API ----------
    artifacts = {
        "feature_engineer": fe,
        "scaler": scaler,
        "feature_order": feature_list,
        "rf": rf_final,
        "xgb": xgb_model,
        "lgb": lgb_model,
        "iso": iso,
        "weights": weights,
        "threshold": float(opt_threshold_final),
        "catboost": cb_model,  # None if not installed
    }
    joblib.dump(artifacts, MODEL_DIR / "ensemble_artifacts.pkl")

    config = {
        "feature_order": feature_list,
        "weights": weights,
        "threshold": opt_threshold_final,
        "risk_scale": 100,
    }
    with open(MODEL_DIR / "ensemble_config.json", "w") as f:
        json.dump(config, f, indent=2)

    joblib.dump(fe, MODEL_DIR / "feature_engineer.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    print(f"\nArtifacts saved to {MODEL_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
