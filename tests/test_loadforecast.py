import numpy as np
import pandas as pd

from loadforecast.models import conformal_quantile, features, fit_ridge, predict, rolling_quantiles


def test_features_use_only_lags_of_24h_or_more():
    idx = pd.date_range("2020-01-01", periods=400, freq="1h")
    s = pd.Series(np.arange(400.0), index=idx)
    X = features(s)
    assert X["lag24"].iloc[100] == 76.0 and X["lag168"].iloc[200] == 32.0
    # perturbing the value at t must not change the features at t
    s2 = s.copy()
    s2.iloc[300] += 1000
    assert features(s2).iloc[300].equals(X.iloc[300])


def test_ridge_recovers_linear_relation():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 3 + rng.normal(0, 0.01, 500)
    w = fit_ridge(X, y, lam=1e-6)
    assert np.abs(predict(w, X) - y).max() < 0.1


def test_conformal_quantile_rank():
    assert conformal_quantile(np.arange(1.0, 100.0), 0.1) == 90.0


def test_rolling_quantiles_use_only_past():
    r = np.concatenate([np.ones(50), 100 * np.ones(50)])
    q = rolling_quantiles(r, 30, 0.1)
    assert np.isnan(q[:30]).all() and q[50] == 1.0 and q[99] == 100.0
