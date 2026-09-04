"""
src/features/build_features.py — Feature engineering pipeline.

Applies temporal, weather, historical fire, and static features to
train.parquet, validation.parquet, and test.parquet.

Usage:
    python -m src.features.build_features

Outputs:
    data/processed/train.parquet       (overwritten with engineered features)
    data/processed/validation.parquet  (overwritten with engineered features)
    data/processed/test.parquet        (overwritten with engineered features)
"""

import logging

import pandas as pd

from src.config import (
    FEATURES_TRAIN,
    FEATURES_VAL,
    FEATURES_TEST,
    FIRES_HARMONIZED,
    INEGI_MUNICIPALITIES,
)
from src.data.weather import load_municipality_catalog
from src.features.temporal import add_temporal_features
from src.features.weather import add_weather_features
from src.features.historical import add_historical_features
from src.features.static import (
    build_static_features,
    add_static_features,
    one_hot_encode,
    CATEGORICAL_COLS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def process_split(
    df: pd.DataFrame,
    harmonized: pd.DataFrame,
    static: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """Apply feature engineering pipeline to a single split."""
    log.info("Processing feature engineering for %s (%d rows)...", split_name, len(df))

    # 1. Temporal features
    df = add_temporal_features(df)

    # 2. Weather features
    df = add_weather_features(df)

    # 3. Historical fire features (zero leakage)
    df = add_historical_features(df, harmonized)

    # 4. Static features
    df = add_static_features(df, static)

    # 5. One-hot encode categoricals
    df = one_hot_encode(df, CATEGORICAL_COLS, drop_first=False)

    log.info("%s feature engineering complete: %d rows | %d columns",
             split_name, len(df), len(df.columns))
    return df


def main() -> None:
    log.info("Loading inputs...")
    harmonized = pd.read_parquet(FIRES_HARMONIZED)
    catalog = load_municipality_catalog()
    static = build_static_features(harmonized, catalog)

    for path, name in [(FEATURES_TRAIN, "Train"), (FEATURES_VAL, "Validation"), (FEATURES_TEST, "Test")]:
        if not path.exists():
            log.warning("Split file not found: %s — skipping", path)
            continue
        split_df = pd.read_parquet(path)
        processed = process_split(split_df, harmonized, static, name)
        processed.to_parquet(path, index=False)
        log.info("Saved engineered features for %s to %s", name, path)

    log.info("Feature engineering complete across all splits.")


if __name__ == "__main__":
    main()
