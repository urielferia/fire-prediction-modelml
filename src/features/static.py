"""
src/features/static.py — Static geographic features per municipality.

Derives features that do not change over time:
  - Bioclimatic zone (backfilled from 2025 dataset)
  - Modal vegetation type (most frequent in historical fires)
  - Region
  - State code
  - Municipality centroid lat/lon
  - Elevation (from Open-Meteo API response if available)

Also performs one-hot encoding of categorical features.
"""

from __future__ import annotations

import logging

import pandas as pd
import numpy as np

log = logging.getLogger(__name__)


def build_static_features(
    harmonized: pd.DataFrame,
    catalog: pd.DataFrame,
) -> pd.DataFrame:
    """Build a static feature table keyed by cvegeo_inegi.

    Parameters
    ----------
    harmonized : pd.DataFrame
        Unified fire-event DataFrame from harmonize.py.
    catalog : pd.DataFrame
        Municipality catalog with centroids (from weather.py).

    Returns
    -------
    pd.DataFrame
        One row per municipality, with static features.
    """
    log.info("Building static municipality features...")

    # Modal categorical values per municipality from fire records
    def modal(s: pd.Series) -> str:
        m = s.dropna().mode()
        return m.iloc[0] if not m.empty else pd.NA

    static = (
        harmonized.dropna(subset=["cvegeo_inegi"])
        .groupby("cvegeo_inegi")
        .agg(
            estado=("estado", modal),
            municipio=("municipio", modal),
            region=("region", modal),
            zona_bioclimatica=("zona_bioclimatica", modal),
            modal_vegetation=("tipo_vegetacion", modal),
            cve_ent=("cve_ent", "first"),
        )
        .reset_index()
    )

    # Join centroid lat/lon and elevation from catalog
    if catalog is not None and not catalog.empty:
        cols_from_catalog = ["cvegeo_inegi", "lat_centroid", "lon_centroid"]
        if "elevation" in catalog.columns:
            cols_from_catalog.append("elevation")
        static = static.merge(
            catalog[cols_from_catalog],
            on="cvegeo_inegi",
            how="left",
        )
    else:
        log.warning("No municipality catalog provided — centroid lat/lon will be missing.")
        static["lat_centroid"] = np.nan
        static["lon_centroid"] = np.nan

    log.info("Static features: %d municipalities", len(static))
    return static


def add_static_features(
    grid: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:
    """Join static municipality features onto the municipality-day grid.

    Parameters
    ----------
    grid : pd.DataFrame
        Municipality-day table.
    static : pd.DataFrame
        Output of build_static_features().

    Returns
    -------
    pd.DataFrame
        Grid with static feature columns appended.
    """
    return grid.merge(static, on="cvegeo_inegi", how="left")


def one_hot_encode(
    df: pd.DataFrame,
    categorical_cols: list[str],
    drop_first: bool = True,
) -> pd.DataFrame:
    """One-hot encode specified categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    categorical_cols : list[str]
        Columns to encode. Missing columns are silently skipped.
    drop_first : bool
        Drop the first category to avoid multicollinearity (for
        Logistic Regression). Keep all for tree models.

    Returns
    -------
    pd.DataFrame
        DataFrame with original columns replaced by OHE dummies.
    """
    present = [c for c in categorical_cols if c in df.columns]
    missing = [c for c in categorical_cols if c not in df.columns]
    if missing:
        log.warning("Columns not found for OHE (skipped): %s", missing)

    if not present:
        return df

    dummies = pd.get_dummies(df[present], drop_first=drop_first, dtype="int8")
    df = df.drop(columns=present)
    df = pd.concat([df, dummies], axis=1)
    return df


# Categorical columns to one-hot encode
CATEGORICAL_COLS = [
    "zona_bioclimatica",
    "modal_vegetation",
    "region",
    "season",         # from temporal.py
]

# These should NOT be one-hot encoded (identifiers or already numeric)
NON_FEATURE_COLS = [
    "cvegeo_inegi",
    "date",
    "n_fires",
    "n_fires_on_day",
    "estado",
    "municipio",
    "dataset",
]

# Target column
TARGET_COL = "fire_occurred"
