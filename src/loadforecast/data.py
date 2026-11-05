from __future__ import annotations

import pandas as pd


def load_hourly(path: str) -> pd.Series:
    """UCI household power consumption -> hourly mean of Global_active_power (kW).

    Hours with fewer than 45 valid minutes are set to NaN (the file has '?' gaps).
    """
    df = pd.read_csv(
        path,
        sep=";",
        na_values="?",
        low_memory=False,
        usecols=["Date", "Time", "Global_active_power"],
    )
    ts = pd.to_datetime(df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S")
    s = pd.Series(df["Global_active_power"].to_numpy(float), index=ts)
    g = s.resample("1h")
    out = g.mean()
    out[g.count() < 45] = float("nan")
    return out
