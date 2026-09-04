"""
src/features/temporal.py — Calendar and seasonal features.

All features are based on the date alone and are always available at
prediction time without any leakage risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Mexico fire season definitions (months)
_SEASON_MAP = {
    1: "dry_cool",    # January — cool dry season
    2: "dry_cool",
    3: "dry_hot",     # March–May — hot dry season (peak fire season)
    4: "dry_hot",
    5: "dry_hot",
    6: "transition",  # June — onset of rainy season
    7: "wet",         # July–September — rainy season
    8: "wet",
    9: "wet",
    10: "transition", # October — end of rainy season
    11: "dry_cool",
    12: "dry_cool",
}

_SEASON_CODE = {
    "dry_cool": 0,
    "transition": 1,
    "dry_hot": 2,
    "wet": 3,
}


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar and seasonal features to a municipality-day DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a date column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with new temporal feature columns appended.
    """
    dt = pd.to_datetime(df[date_col])

    # ── Raw calendar ──────────────────────────────────────────────────────────
    df["month"] = dt.dt.month.astype("int8")
    df["day_of_year"] = dt.dt.dayofyear.astype("int16")
    df["week_of_year"] = dt.dt.isocalendar().week.astype("int8")
    df["day_of_week"] = dt.dt.dayofweek.astype("int8")     # 0=Monday
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype("int8")
    df["year"] = dt.dt.year.astype("int16")

    # ── Cyclical encoding (preserves circular nature of calendar) ─────────────
    df["sin_day_of_year"] = np.sin(2 * np.pi * dt.dt.dayofyear / 365.0)
    df["cos_day_of_year"] = np.cos(2 * np.pi * dt.dt.dayofyear / 365.0)
    df["sin_month"] = np.sin(2 * np.pi * dt.dt.month / 12.0)
    df["cos_month"] = np.cos(2 * np.pi * dt.dt.month / 12.0)
    df["sin_week"] = np.sin(2 * np.pi * dt.dt.isocalendar().week.astype(int) / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * dt.dt.isocalendar().week.astype(int) / 52.0)

    # ── Season ────────────────────────────────────────────────────────────────
    df["season"] = dt.dt.month.map(_SEASON_MAP)
    df["season_code"] = df["season"].map(_SEASON_CODE).astype("int8")

    # ── Fire season flag ──────────────────────────────────────────────────────
    # Peak fire season: February–May (months 2–5)
    df["is_fire_season"] = dt.dt.month.between(2, 5).astype("int8")

    return df
