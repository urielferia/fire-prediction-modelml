"""
src/features/weather.py — Weather feature engineering from Open-Meteo daily data.

Computes same-day weather values, lagged values, and rolling statistics.
All features use only past/current data — no future information leakage.

Note: same-day weather values use ERA5 reanalysis in training, which
approximates operational day-ahead forecasts available at prediction time.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Raw weather columns from Open-Meteo
_RAW_WEATHER = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "windspeed_10m_max",
    "windgusts_10m_max",
    "et0_fao_evapotranspiration",
    "shortwave_radiation_sum",
    "relative_humidity_2m_mean",
]


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add weather-derived features to a municipality-day DataFrame.

    Requires:
        - All columns in _RAW_WEATHER to be present.
        - df sorted by (cvegeo_inegi, date).

    Parameters
    ----------
    df : pd.DataFrame
        Municipality-day table, already joined with weather data.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new weather feature columns appended.
    """
    df = df.sort_values(["cvegeo_inegi", "date"]).copy()

    # ── Validate raw columns ──────────────────────────────────────────────────
    missing = [c for c in _RAW_WEATHER if c not in df.columns]
    if missing:
        log.warning("Missing raw weather columns: %s — they will be NaN", missing)

    # ── Convenience aliases (shorter names) ───────────────────────────────────
    df["temp_max"] = df.get("temperature_2m_max")
    df["temp_min"] = df.get("temperature_2m_min")
    df["temp_mean"] = df.get("temperature_2m_mean")
    df["precip"] = df.get("precipitation_sum")
    df["rain"] = df.get("rain_sum")
    df["wind_max"] = df.get("windspeed_10m_max")
    df["wind_gusts"] = df.get("windgusts_10m_max")
    df["et0"] = df.get("et0_fao_evapotranspiration")
    df["radiation"] = df.get("shortwave_radiation_sum")
    df["humidity"] = df.get("relative_humidity_2m_mean")

    # ── Derived same-day features ─────────────────────────────────────────────
    df["temp_range"] = df["temp_max"] - df["temp_min"]   # daily temperature range
    df["is_dry_day"] = (df["precip"] < 1.0).astype("int8")   # <1mm → dry

    # ── Per-municipality groupby for time-series features ─────────────────────
    grp = df.groupby("cvegeo_inegi", sort=False)

    # Lagged temperature (previous 1, 2, 3 days)
    for lag in [1, 2, 3]:
        df[f"temp_max_lag{lag}"] = grp["temp_max"].shift(lag)

    # Lagged precipitation (previous 1–7 days)
    for lag in range(1, 8):
        df[f"precip_lag{lag}"] = grp["precip"].shift(lag)

    # Rolling sums of precipitation (past 7, 14, 30 days)
    # shift(1) ensures we use only past data (not the current day)
    precip_shifted = grp["precip"].shift(1)
    df["precip_sum_7d"] = grp["precip"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).sum()
    )
    df["precip_sum_14d"] = grp["precip"].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).sum()
    )
    df["precip_sum_30d"] = grp["precip"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).sum()
    )

    # Rolling mean of max temperature (past 7 and 30 days)
    df["temp_max_7d_mean"] = grp["temp_max"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    df["temp_max_30d_mean"] = grp["temp_max"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).mean()
    )

    # Rolling max of wind gusts (past 3 days)
    df["wind_gusts_3d_max"] = grp["wind_gusts"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).max()
    )

    # Consecutive dry days streak (days since last precipitation ≥ 1mm)
    # Computed per municipality group
    df["dry_streak"] = grp["is_dry_day"].transform(_consecutive_dry_days)

    # Rolling mean of relative humidity (past 7 days)
    df["humidity_7d_mean"] = grp["humidity"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )

    # ET0 rolling sum (past 7 days — cumulative evapotranspiration)
    df["et0_7d_sum"] = grp["et0"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).sum()
    )

    return df


def _consecutive_dry_days(series: pd.Series) -> pd.Series:
    """Compute consecutive dry days ending at each date (exclusive of current day).

    A 'dry day' is defined as is_dry_day == 1 (precipitation < 1mm).
    Uses the shifted series so only past data is included.
    """
    shifted = series.shift(1)
    result = []
    streak = 0
    for val in shifted:
        if pd.isna(val):
            result.append(np.nan)
        elif val == 1:
            streak += 1
            result.append(streak)
        else:
            streak = 0
            result.append(0)
    return pd.Series(result, index=series.index)
