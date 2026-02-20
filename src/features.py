"""
Feature engineering for fraud detection: amount buckets, time-based features, transaction velocity.
Shared between training and inference (app).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# Base feature columns used by the original model (before engineered features)
BASE_FEATURE_ORDER = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds engineered features: amount buckets, time-based features, transaction velocity.
    Fits on training data (velocity stats, amount quantiles); at inference uses stored stats.
    """

    def __init__(self, amount_quantiles=(0.33, 0.66), velocity_window_sec=3600):
        self.amount_quantiles = amount_quantiles
        self.velocity_window_sec = velocity_window_sec
        self.amount_bounds_ = None  # (low, high) for 3 buckets
        self.velocity_median_ = None  # fallback when velocity cannot be computed at inference

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            time_col = X["Time"].values
            amount_col = X["Amount"].values
        else:
            # X is array; assume order is BASE_FEATURE_ORDER
            time_col = X[:, 0]
            amount_col = X[:, -1]
        amount_col = np.asarray(amount_col, dtype=float).ravel()
        q = np.nanpercentile(amount_col, [p * 100 for p in self.amount_quantiles])
        self.amount_bounds_ = (float(q[0]), float(q[1]))
        # Velocity: transactions per window. Compute per-row then take median for inference fallback.
        velocity = self._compute_velocity(time_col)
        self.velocity_median_ = float(np.nanmedian(velocity)) if velocity.size else 0.0
        return self

    def _compute_velocity(self, time_arr):
        """Transactions in the last velocity_window_sec (for each row)."""
        time_arr = np.asarray(time_arr, dtype=float).ravel()
        n = len(time_arr)
        out = np.zeros(n)
        if n == 0:
            return out
        order = np.argsort(time_arr)
        sorted_t = time_arr[order]
        for i in range(n):
            t = sorted_t[i]
            window_start = t - self.velocity_window_sec
            count = np.searchsorted(sorted_t, window_start, side="right")
            # count of points in (window_start, t] is i - count + 1 (including self)
            out[order[i]] = (i - count + 1)
        return out

    def transform(self, X, velocity_override=None):
        """
        Add engineered features. If velocity_override is provided (single row inference),
        use it; otherwise use velocity_median_ from fit.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            time_col = df["Time"].values
            amount_col = df["Amount"].values
            base = df[BASE_FEATURE_ORDER] if all(c in df.columns for c in BASE_FEATURE_ORDER) else df
        else:
            X = np.asarray(X)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            time_col = X[:, 0]
            amount_col = X[:, -1]
            base = pd.DataFrame(X, columns=BASE_FEATURE_ORDER)
            df = base.copy()

        # Amount buckets (0=low, 1=mid, 2=high)
        low, high = self.amount_bounds_
        amount_bucket = np.ones(len(amount_col), dtype=int)
        amount_bucket[amount_col <= low] = 0
        amount_bucket[amount_col > high] = 2
        df["amount_bucket"] = amount_bucket

        # Time-based: hour of day (0-23) from seconds; cyclic encoding
        time_sec = np.asarray(time_col, dtype=float)
        hour = (time_sec // 3600) % 24
        df["hour_of_day"] = hour
        df["time_sin"] = np.sin(2 * np.pi * hour / 24)
        df["time_cos"] = np.cos(2 * np.pi * hour / 24)

        # Transaction velocity
        if velocity_override is not None:
            vel = np.broadcast_to(velocity_override, (len(time_col),))
        elif hasattr(self, "velocity_median_") and self.velocity_median_ is not None:
            vel = np.full(len(time_col), self.velocity_median_, dtype=float)
        else:
            vel = self._compute_velocity(time_col)
        df["velocity_1h"] = vel

        return df

    def get_feature_names_out(self, input_features=None):
        base = BASE_FEATURE_ORDER.copy()
        extra = ["amount_bucket", "hour_of_day", "time_sin", "time_cos", "velocity_1h"]
        return np.array(base + extra)


def get_feature_order():
    """Order of columns after feature engineering (for model input)."""
    return BASE_FEATURE_ORDER + ["amount_bucket", "hour_of_day", "time_sin", "time_cos", "velocity_1h"]
