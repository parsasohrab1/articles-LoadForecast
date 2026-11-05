# Software Requirements Specification – LoadForecast

- SR-1 `data.load_hourly`: hourly mean of Global_active_power (kW); hours with fewer than 45 valid minutes are missing.
- SR-2 `models.features`: lags 24, 48, 168 h, mean of the previous 24 h (all shifted by at least 24 h), hour-of-day dummies, weekend flag, annual sine/cosine. A unit test checks that changing the value at time t does not change the features at t.
- SR-3 `models.fit_ridge`/`predict`: closed-form ridge regression.
- SR-4 `models.conformal_quantile`: split-conformal quantile of absolute residuals; `rolling_quantiles`: the same using only the previous N residuals (past only).
- SR-5 `research/run_study.py`: train 2007; calibration 2008-01..06; test 2008-07 to 2010-11; baselines seasonal naive 24 h and 168 h; fixed vs rolling (90-day window) interval.
- SR-6 Four unit tests.
