"""
src/features/historical.py — Historical fire frequency features.

IMPORTANT — Leakage prevention:
    All features for a prediction on date D use only fires with
    fecha_inicio < D. Current-day fires are never included.

Features are computed from the harmonized fire dataset and joined
onto the municipality-day grid.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def add_historical_features(
    grid: pd.DataFrame,
    harmonized: pd.DataFrame,
) -> pd.DataFrame:
    """Add historical fire frequency features to the municipality-day grid.

    Parameters
    ----------
    grid : pd.DataFrame
        Municipality-day table (output of build_dataset.py).
        Must have columns: cvegeo_inegi, date.
    harmonized : pd.DataFrame
        Full harmonized fire-event DataFrame (output of harmonize.py).
        Must have columns: cvegeo_inegi, fecha_inicio.

    Returns
    -------
    pd.DataFrame
        Grid with historical fire feature columns appended.
    """
    log.info("Computing historical fire features...")

    grid = grid.copy()
    grid["date"] = pd.to_datetime(grid["date"])
    harmonized = harmonized.dropna(subset=["cvegeo_inegi", "fecha_inicio"]).copy()
    harmonized["fecha_inicio"] = pd.to_datetime(harmonized["fecha_inicio"])

    # Build a daily fire count series per municipality from fire records
    # (this is the source of truth for all historical features)
    daily_fires = (
        harmonized.groupby(["cvegeo_inegi", harmonized["fecha_inicio"].dt.normalize()])
        .size()
        .reset_index(name="n_fires_on_day")
        .rename(columns={"fecha_inicio": "date"})
    )
    daily_fires["date"] = pd.to_datetime(daily_fires["date"])

    # Merge daily fires onto the full grid — NaN means 0 fires that day
    grid = grid.merge(daily_fires, on=["cvegeo_inegi", "date"], how="left")
    grid["n_fires_on_day"] = grid["n_fires_on_day"].fillna(0).astype("int16")

    # ── Per-municipality rolling historical features ───────────────────────────
    log.info("Computing rolling fire counts per municipality...")

    grid = grid.sort_values(["cvegeo_inegi", "date"]).reset_index(drop=True)

    grp = grid.groupby("cvegeo_inegi", sort=False)

    # Fires in previous 7 days (strictly before current date: shift then roll)
    grid["fires_7d"] = grp["n_fires_on_day"].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).sum()
    ).astype("float32")

    # Fires in previous 30 days
    grid["fires_30d"] = grp["n_fires_on_day"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).sum()
    ).astype("float32")

    # Days since last fire in this municipality
    grid["days_since_last_fire"] = grp["n_fires_on_day"].transform(
        _days_since_last_fire
    ).astype("float32")

    # ── Cross-municipality regional features ──────────────────────────────────
    log.info("Computing regional fire pressure features...")

    # Map CVEGEO to state code (first 2 digits of 5-digit code)
    grid["cve_ent_str"] = grid["cvegeo_inegi"].astype(str).str.zfill(5).str[:2]

    # Daily fire counts by state
    state_daily = (
        grid.groupby(["cve_ent_str", "date"])["n_fires_on_day"]
        .sum()
        .reset_index(name="state_fires_today")
    )
    state_daily["date"] = pd.to_datetime(state_daily["date"])
    grid = grid.merge(state_daily, on=["cve_ent_str", "date"], how="left")

    # State fires in previous 30 days
    grid = grid.sort_values(["cve_ent_str", "date"]).reset_index(drop=True)
    state_grp = grid.groupby("cve_ent_str", sort=False)
    grid["fires_state_30d"] = state_grp["state_fires_today"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=1).sum()
    ).astype("float32")

    # Clean up temp columns
    grid = grid.drop(columns=["state_fires_today", "cve_ent_str"], errors="ignore")

    # ── Annual historical fire rate per municipality ───────────────────────────
    log.info("Computing annual historical fire rate per municipality...")

    annual_rate = _compute_annual_fire_rate(harmonized)
    grid = grid.merge(annual_rate, on="cvegeo_inegi", how="left")
    grid["annual_fire_rate"] = grid["annual_fire_rate"].fillna(0).astype("float32")

    # ── Same-month historical fire average ────────────────────────────────────
    log.info("Computing same-month historical averages per municipality...")

    monthly_avg = _compute_monthly_avg(harmonized)
    grid["month"] = pd.to_datetime(grid["date"]).dt.month.astype("int8")
    grid = grid.merge(monthly_avg, on=["cvegeo_inegi", "month"], how="left")
    grid["avg_fires_same_month"] = grid["avg_fires_same_month"].fillna(0).astype("float32")

    log.info("Historical features complete.")
    return grid


# ── Helper functions ──────────────────────────────────────────────────────────

def _days_since_last_fire(series: pd.Series) -> pd.Series:
    """For each position, count days since the most recent non-zero value.

    Uses shift(1) to exclude the current day.
    Returns NaN if no prior fire exists.
    """
    shifted = series.shift(1)
    result = []
    days = np.nan
    for val in shifted:
        if pd.isna(val):
            result.append(np.nan)
        elif val > 0:
            days = 0
            result.append(days)
        else:
            if not np.isnan(days):
                days += 1
            result.append(days)
    return pd.Series(result, index=series.index, dtype="float32")


def _compute_annual_fire_rate(harmonized: pd.DataFrame) -> pd.DataFrame:
    """Compute average annual fire count per municipality across all years."""
    yearly = (
        harmonized.assign(anio=harmonized["fecha_inicio"].dt.year)
        .groupby(["cvegeo_inegi", "anio"])
        .size()
        .reset_index(name="fires_in_year")
    )
    rate = (
        yearly.groupby("cvegeo_inegi")["fires_in_year"]
        .mean()
        .reset_index(name="annual_fire_rate")
    )
    return rate


def _compute_monthly_avg(harmonized: pd.DataFrame) -> pd.DataFrame:
    """Compute average monthly fire count per municipality (across all years)."""
    monthly = (
        harmonized.assign(month=harmonized["fecha_inicio"].dt.month)
        .groupby(["cvegeo_inegi", "month"])
        .size()
        .reset_index(name="fires_in_month_year")
    )
    # Average across years
    avg = (
        monthly.groupby(["cvegeo_inegi", "month"])["fires_in_month_year"]
        .mean()
        .reset_index(name="avg_fires_same_month")
    )
    return avg
