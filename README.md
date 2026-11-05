# articles-LoadForecast

Day-ahead hourly electricity-load forecasts with conformal prediction intervals, evaluated chronologically on real data
(UCI Individual Household Electric Power Consumption: one household, 2006–2010). TRL 4 (public real data, laboratory validation only).

```
pip install -e ".[dev]" && python -m pytest
# data: https://archive.ics.uci.edu/dataset/235 (not committed)
python research/run_study.py /path/to/household_power_consumption.txt
```
