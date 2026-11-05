from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (24, 48, 168)


def features(s: pd.Series) -> pd.DataFrame:
    """Day-ahead features: every lag is >= 24 h, so only information known the day before is used."""
    X = pd.DataFrame(index=s.index)
    for lag in LAGS:
        X[f"lag{lag}"] = s.shift(lag)
    X["mean_prev24"] = s.shift(24).rolling(24, min_periods=18).mean()
    h = s.index.hour
    for k in range(24):
        X[f"h{k}"] = (h == k).astype(float)
    X["weekend"] = (s.index.dayofweek >= 5).astype(float)
    X["sin_y"] = np.sin(2 * np.pi * s.index.dayofyear / 365.25)
    X["cos_y"] = np.cos(2 * np.pi * s.index.dayofyear / 365.25)
    return X


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def conformal_quantile(res: np.ndarray, alpha: float) -> float:
    n = len(res)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    s = np.sort(res)
    return float(s[min(k, n) - 1])


def rolling_quantiles(abs_res: np.ndarray, window: int, alpha: float) -> np.ndarray:
    """Rolling recalibration: half-width at step t from the previous ``window`` residuals only."""
    out = np.full(len(abs_res), np.nan)
    for t in range(window, len(abs_res)):
        out[t] = conformal_quantile(abs_res[t - window : t], alpha)
    return out
