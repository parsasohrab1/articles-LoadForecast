"""UCI household power consumption (one French household, 2006-2010): day-ahead hourly forecasts.

usage: python research/run_study.py /path/to/household_power_consumption.txt
train 2007; calibration 2008-01..2008-06; test 2008-07..2010-11 (chronological).
Fixed split-conformal half-width vs rolling recalibration (last 90 days of test residuals).
"""

import json
import sys
from pathlib import Path

import numpy as np

from loadforecast.data import load_hourly
from loadforecast.models import conformal_quantile, features, fit_ridge, predict, rolling_quantiles

ALPHA = 0.1
WINDOW = 24 * 90


def main(path: str):
    s = load_hourly(path)
    X = features(s)
    ok = X.notna().all(axis=1) & s.notna()
    d = s[ok]
    Xd = X[ok]
    tr = (d.index >= "2007-01-01") & (d.index < "2008-01-01")
    ca = (d.index >= "2008-01-01") & (d.index < "2008-07-01")
    te = d.index >= "2008-07-01"
    w = fit_ridge(Xd[tr].to_numpy(), d[tr].to_numpy())
    pred = predict(w, Xd.to_numpy())
    y = d.to_numpy()
    naive24 = Xd["lag24"].to_numpy()
    naive168 = Xd["lag168"].to_numpy()
    res = np.abs(y - pred)
    out = {
        "note": "real data, one household; hourly mean kW; day-ahead (features use lags >= 24 h); "
        "no weather or occupancy inputs",
        "n_train": int(tr.sum()),
        "n_cal": int(ca.sum()),
        "n_test": int(te.sum()),
        "test_period": [str(d.index[te][0]), str(d.index[te][-1])],
        "mae_test_kw": {
            "ridge": float(res[te].mean()),
            "naive_24h": float(np.abs(y - naive24)[te].mean()),
            "naive_168h": float(np.abs(y - naive168)[te].mean()),
        },
        "mean_load_test_kw": float(y[te].mean()),
    }
    q_fixed = conformal_quantile(res[ca], ALPHA)
    cov_fixed = res <= q_fixed
    # rolling recalibration on the chronological sequence of test residuals (past only)
    q_roll = rolling_quantiles(res[te], WINDOW, ALPHA)
    cov_roll = res[te] <= q_roll
    valid = ~np.isnan(q_roll)
    months = d.index[te].to_period("M")
    by_month = {}
    for m in sorted(set(months)):
        mm = np.asarray(months == m)
        if mm.sum() < 100:
            continue
        by_month[str(m)] = {
            "fixed_cov": float(cov_fixed[te][mm].mean()),
            "rolling_cov": float(cov_roll[mm & valid].mean()) if (mm & valid).any() else None,
        }
    fx = np.array([v["fixed_cov"] for v in by_month.values()])
    out["alpha"] = ALPHA
    out["fixed_split"] = {
        "halfwidth_kw": q_fixed,
        "coverage_calibration": float(cov_fixed[ca].mean()),
        "coverage_test": float(cov_fixed[te].mean()),
        "monthly_cov_min": float(fx.min()),
        "monthly_cov_max": float(fx.max()),
        "months_below_0.8": int((fx < 0.8).sum()),
        "n_months": len(fx),
    }
    out["rolling_recalibration"] = {
        "window_hours": WINDOW,
        "coverage_test": float(cov_roll[valid].mean()),
        "mean_halfwidth_kw": float(q_roll[valid].mean()),
    }
    rl = [v["rolling_cov"] for v in by_month.values() if v["rolling_cov"] is not None]
    out["rolling_recalibration"].update(
        {"monthly_cov_min": float(min(rl)), "monthly_cov_max": float(max(rl))}
    )
    out["monthly"] = by_month
    Path(__file__).with_name("results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "monthly"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
