"""
src/config.py — Central configuration: paths, constants, random seeds.

All other modules should import from here rather than hardcoding values.
"""

import os
from datetime import date, timedelta
from pathlib import Path

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Directory layout ─────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Ensure output directories exist at import time
for _d in [INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Raw dataset filenames ─────────────────────────────────────────────────────
RAW_2015_2024 = RAW_DIR / "estadisticasincendiosforestales2015-2024.csv"
RAW_2025 = RAW_DIR / "2025_Incendios_forestales.csv"

# ── External dataset filenames ────────────────────────────────────────────────
INEGI_MUNICIPALITIES = EXTERNAL_DIR / "inegi_municipalities.csv"
WEATHER_DIR = EXTERNAL_DIR / "weather"

# ── Interim / processed filenames ────────────────────────────────────────────
FIRES_HARMONIZED = INTERIM_DIR / "fires_harmonized.parquet"
MUNICIPALITY_DAYS = INTERIM_DIR / "municipality_days.parquet"
WEATHER_DAILY = INTERIM_DIR / "weather_daily.parquet"
FEATURES_TRAIN = PROCESSED_DIR / "train.parquet"
FEATURES_VAL = PROCESSED_DIR / "validation.parquet"
FEATURES_TEST = PROCESSED_DIR / "test.parquet"

# ── Temporal split boundaries ─────────────────────────────────────────────────
TRAIN_START = "2016-01-01"   # 2015 excluded pending completeness check
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2025-12-31"

# ── Sampling ─────────────────────────────────────────────────────────────────
# 20:1 negative-to-positive ratio for training (team decision)
NEGATIVE_POSITIVE_RATIO = 20

# ── Mexico geographic bounds (for coordinate validation) ─────────────────────
MEX_LAT_MIN = 14.5
MEX_LAT_MAX = 32.7
MEX_LON_MIN = -118.5
MEX_LON_MAX = -86.7

# ── Risk threshold percentiles (team decision: quantile-based) ────────────────
# Applied to predicted probabilities across all validation municipality-days
MEDIUM_RISK_PERCENTILE = 90   # above this → MEDIUM
HIGH_RISK_PERCENTILE = 99     # above this → HIGH

# ── Open-Meteo API ───────────────────────────────────────────────────────────
# ERA5 reanalysis archive (historical, up to ~5 days ago)
OPENMETEO_URL = "https://archive-api.open-meteo.com/v1/archive"
# Weather forecast API (up to 16 days ahead, overlaps recent past by ~5 days)
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_VARIABLES = [
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
WEATHER_FETCH_START = "2014-11-01"  # extra buffer for rolling features
# Dynamic end date — keeps bulk downloads up to date
WEATHER_FETCH_END = date.today().strftime("%Y-%m-%d")

# ── Forecast horizon ──────────────────────────────────────────────────────────
# Max days into the future that users can query (Open-Meteo forecast supports ~16)
MAX_FORECAST_DAYS = 15
# ERA5 reanalysis lags real-time by ~5 days; dates newer than this use forecast API
ERA5_LAG_DAYS = 6

# ── Model filenames ───────────────────────────────────────────────────────────
MODEL_LR = MODELS_DIR / "logistic_regression.joblib"
MODEL_RF = MODELS_DIR / "random_forest.joblib"
MODEL_LGBM = MODELS_DIR / "lightgbm.joblib"
CALIBRATOR = MODELS_DIR / "calibrator.joblib"
PREPROCESSOR = MODELS_DIR / "preprocessor.joblib"
THRESHOLDS = MODELS_DIR / "risk_thresholds.json"
