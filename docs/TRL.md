# TRL assessment – LoadForecast

**Claimed level: TRL 4** (laboratory validation on one public real dataset). Not TRL 5: one household, no weather or occupancy data, no operational environment.

Data: UCI Individual Household Electric Power Consumption, hourly means; train 8570 h, calibration 4368 h, test 20069 h (2008-07-01 to 2010-11-26). Mean test load 1.05 kW.

| Forecast | Test MAE (kW) |
|---|---|
| Ridge (day-ahead features) | 0.490 |
| Naive, same hour yesterday | 0.580 |
| Naive, same hour last week | 0.573 |

90 % prediction intervals (half-width in kW):

| Interval | Half-width | Coverage on test | Monthly coverage range |
|---|---|---|---|
| Fixed split-conformal (calibrated on 2008-01..06) | 1.17 | 0.929 | 0.85 – 0.99 |
| Rolling recalibration (last 90 days) | 1.04 (mean) | 0.898 | 0.79 – 0.95 |

## Findings and caveats
- The ridge model lowers MAE by about 15 % against the best naive baseline (0.49 vs 0.57 kW) – a modest gain for a single household.
- The fixed interval over-covers on the test period (0.929 against 0.90 on calibration) and its monthly coverage varies from 0.85 to 0.99: the half-width calibrated on winter/spring residuals is too wide in summer. No month fell below 0.80.
- Rolling recalibration restores the average coverage (0.898) with a narrower interval (about 11 % smaller) but individual months fall to 0.79, because the window lags behind abrupt changes.
- The interval half-width (about 1.0–1.2 kW) is as large as the mean load (1.05 kW): household load is hard to predict a day ahead without weather or occupancy information, and these intervals have limited operational value.
- Monthly figures are about 692 hours each; coverage guarantees are marginal, not per-hour, and residuals are autocorrelated, so the exchangeability assumption behind split conformal is only approximate.
