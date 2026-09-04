"""
src/data/build_dataset.py — Construct the municipality × day modeling table.

For every fire-active municipality, generates one row per calendar day
in the study period (2016–2025), labels fire-positive days, joins weather
data, and applies 20:1 negative undersampling for the training split.

Usage:
    python -m src.data.build_dataset

Outputs:
    data/interim/municipality_days.parquet  (full labeled table, all years)
    data/processed/train.parquet
    data/processed/validation.parquet
    data/processed/test.parquet
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import (
    FIRES_HARMONIZED,
    MUNICIPALITY_DAYS,
    WEATHER_DAILY,
    FEATURES_TRAIN,
    FEATURES_VAL,
    FEATURES_TEST,
    TRAIN_START,
    TRAIN_END,
    VAL_START,
    VAL_END,
    TEST_START,
    TEST_END,
    NEGATIVE_POSITIVE_RATIO,
    RANDOM_SEED,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Positive labels ───────────────────────────────────────────────────────────

def _build_positive_labels(harmonized: pd.DataFrame) -> pd.DataFrame:
    """Create municipality-day records where fire_occurred = 1.

    Rules:
    - One row per unique (cvegeo_inegi, date), even if multiple fires.
    - Date = fecha_inicio (ignition date only).
    - Aggregated: count of fires, sum of hectares.
    """
    pos = (
        harmonized.dropna(subset=["cvegeo_inegi", "fecha_inicio"])
        .assign(date=lambda df: df["fecha_inicio"].dt.normalize())
        .groupby(["cvegeo_inegi", "date"])
        .agg(
            fire_occurred=("fecha_inicio", "count"),   # will be binarized
            n_fires=("fecha_inicio", "count"),
        )
        .reset_index()
    )
    pos["fire_occurred"] = 1
    return pos


# ── Full municipality-day enumeration ─────────────────────────────────────────

def _enumerate_municipality_days(
    fire_active_muni: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Generate all (municipality, day) combinations for the date range.

    Parameters
    ----------
    fire_active_muni : list[str]
        CVEGEO codes of municipalities that had ≥1 fire.
    start : str
        Start date (inclusive).
    end : str
        End date (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: cvegeo_inegi, date  (all combinations, fire_occurred=0)
    """
    all_dates = pd.date_range(start=start, end=end, freq="D")
    log.info(
        "Enumerating %d municipalities × %d days = %s rows...",
        len(fire_active_muni), len(all_dates),
        f"{len(fire_active_muni) * len(all_dates):,}",
    )

    # Build via cross join
    munis = pd.DataFrame({"cvegeo_inegi": fire_active_muni})
    dates = pd.DataFrame({"date": all_dates})
    munis["_key"] = 1
    dates["_key"] = 1
    grid = munis.merge(dates, on="_key").drop(columns="_key")
    grid["fire_occurred"] = 0
    return grid


# ── Dataset assembly ──────────────────────────────────────────────────────────

def build_municipality_days(
    harmonized: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the full municipality-day table with target labels and weather.

    Parameters
    ----------
    harmonized : pd.DataFrame
        Unified fire-event DataFrame from harmonize.py.
    weather : pd.DataFrame
        Daily weather DataFrame from weather.py.

    Returns
    -------
    pd.DataFrame
        Municipality-day table with target and weather columns.
    """
    # 1. Determine fire-active municipalities
    fire_active = harmonized["cvegeo_inegi"].dropna().unique().tolist()
    log.info("Fire-active municipalities: %d", len(fire_active))

    # 2. Positive labels (from fire records)
    positives = _build_positive_labels(harmonized)
    positives = positives[positives["cvegeo_inegi"].isin(fire_active)]
    log.info("Positive municipality-days: %d", len(positives))

    # 3. Full grid (all municipalities × all days across whole study window)
    # Use 2016-01-01 as start (2015 excluded)
    grid = _enumerate_municipality_days(fire_active, "2016-01-01", TEST_END)

    # 4. Merge positives onto grid (updates fire_occurred where fires occurred)
    grid = grid.merge(
        positives[["cvegeo_inegi", "date", "fire_occurred", "n_fires"]],
        on=["cvegeo_inegi", "date"],
        how="left",
        suffixes=("", "_pos"),
    )
    # Resolve: use positive value where available
    grid["fire_occurred"] = grid["fire_occurred_pos"].fillna(grid["fire_occurred"]).astype("int8")
    grid["n_fires"] = grid["n_fires"].fillna(0).astype("int16")
    grid = grid.drop(columns=["fire_occurred_pos"], errors="ignore")

    log.info(
        "Grid: %s rows | %d positive (%.3f%%)",
        f"{len(grid):,}",
        grid["fire_occurred"].sum(),
        100 * grid["fire_occurred"].mean(),
    )

    # 5. Join weather data
    if weather is not None and not weather.empty:
        weather["date"] = pd.to_datetime(weather["date"])
        weather["cvegeo_inegi"] = weather["cvegeo_inegi"].astype(str)
        grid = grid.merge(weather, on=["cvegeo_inegi", "date"], how="left")
        weather_coverage = grid[OPENMETEO_VARIABLES[0]].notna().mean()
        log.info("Weather join coverage: %.1f%%", 100 * weather_coverage)
    else:
        log.warning("No weather data provided — weather columns will be missing.")

    return grid


# ── Train/val/test splits ─────────────────────────────────────────────────────

def apply_temporal_splits(grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split grid into train / validation / test by date.

    Training set also applies 20:1 negative undersampling.
    """
    train_mask = grid["date"].between(TRAIN_START, TRAIN_END)
    val_mask = grid["date"].between(VAL_START, VAL_END)
    test_mask = grid["date"].between(TEST_START, TEST_END)

    train = grid[train_mask].copy()
    val = grid[val_mask].copy()
    test = grid[test_mask].copy()

    log.info("Train: %s rows | %.3f%% positive", f"{len(train):,}", 100 * train["fire_occurred"].mean())
    log.info("Val  : %s rows | %.3f%% positive", f"{len(val):,}", 100 * val["fire_occurred"].mean())
    log.info("Test : %s rows | %.3f%% positive", f"{len(test):,}", 100 * test["fire_occurred"].mean())

    # Apply 20:1 undersampling to training set only
    train = _undersample_negatives(train, ratio=NEGATIVE_POSITIVE_RATIO)
    log.info("Train after 20:1 undersampling: %s rows | %.2f%% positive",
             f"{len(train):,}", 100 * train["fire_occurred"].mean())

    return train, val, test


def _undersample_negatives(df: pd.DataFrame, ratio: int) -> pd.DataFrame:
    """Keep all positives; randomly sample negatives at `ratio` × n_positives."""
    rng = np.random.default_rng(RANDOM_SEED)

    pos = df[df["fire_occurred"] == 1]
    neg = df[df["fire_occurred"] == 0]

    n_pos = len(pos)
    n_neg_target = n_pos * ratio
    n_neg_target = min(n_neg_target, len(neg))  # don't exceed available negatives

    neg_sampled = neg.sample(n=n_neg_target, random_state=RANDOM_SEED)

    return pd.concat([pos, neg_sampled]).sort_values("date").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("Loading harmonized fire data...")
    if not FIRES_HARMONIZED.exists():
        raise FileNotFoundError(
            f"fires_harmonized.parquet not found. Run src.data.harmonize first."
        )
    harmonized = pd.read_parquet(FIRES_HARMONIZED)

    log.info("Loading weather data...")
    if not WEATHER_DAILY.exists():
        raise FileNotFoundError(
            f"weather_daily.parquet not found. Run src.data.weather first."
        )
    weather = pd.read_parquet(WEATHER_DAILY)

    log.info("Building municipality-day grid...")
    grid = build_municipality_days(harmonized, weather)

    log.info("Saving full municipality-day table...")
    MUNICIPALITY_DAYS.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(MUNICIPALITY_DAYS, index=False)
    log.info("Saved to %s", MUNICIPALITY_DAYS)

    log.info("Applying temporal splits...")
    train, val, test = apply_temporal_splits(grid)

    train.to_parquet(FEATURES_TRAIN, index=False)
    val.to_parquet(FEATURES_VAL, index=False)
    test.to_parquet(FEATURES_TEST, index=False)
    log.info("Splits saved to %s", FEATURES_TRAIN.parent)


# Import for use in build_municipality_days above
try:
    from src.config import OPENMETEO_VARIABLES
except ImportError:
    OPENMETEO_VARIABLES = []


if __name__ == "__main__":
    main()
