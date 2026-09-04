"""
src/features/risk_cache.py — Precompute daily wildfire risk across all Mexican municipalities.

Generates a unified daily risk lookup table for fast map visualization:
- 2015–2025: Ground truth fires + calibrated LightGBM model predictions
- 2026: Open-Meteo 16-day forecast horizon (today through today + 15 days)

Output:
    data/processed/daily_risk_cache.parquet
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
import urllib.request

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CALIBRATOR,
    THRESHOLDS,
    FEATURES_TRAIN,
    FEATURES_VAL,
    FEATURES_TEST,
    FIRES_HARMONIZED,
    WEATHER_DAILY,
    INEGI_MUNICIPALITIES,
    PROCESSED_DIR,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

CACHE_FILE = PROCESSED_DIR / "daily_risk_cache.parquet"


def _load_model_and_thresholds():
    bundle = joblib.load(CALIBRATOR)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    with open(THRESHOLDS) as f:
        thresholds = json.load(f)

    return model, feature_cols, thresholds


def build_daily_risk_cache() -> pd.DataFrame:
    """Build the unified daily risk cache for 2015 through 2026."""
    log.info("Starting daily risk cache generation...")
    model, feature_cols, thresholds = _load_model_and_thresholds()
    med_thr = thresholds["medium_threshold"]
    high_thr = thresholds["high_threshold"]

    dfs: list[pd.DataFrame] = []

    # ── 1. Process 2025 (Test set: full grid, all 1,737 munis) ───────────────────
    if FEATURES_TEST.exists():
        log.info("Processing 2025 test split...")
        t0 = time.time()
        test_df = pd.read_parquet(
            FEATURES_TEST,
            columns=["cvegeo_inegi", "date", "n_fires"] + feature_cols,
        )
        probs = model.predict_proba(test_df[feature_cols])[:, 1]
        risks = np.zeros(len(probs), dtype=np.int8)
        risks[probs >= med_thr] = 1
        risks[probs >= high_thr] = 2

        dfs.append(pd.DataFrame({
            "cvegeo": test_df["cvegeo_inegi"].astype(str).str.zfill(5),
            "date": pd.to_datetime(test_df["date"]).dt.strftime("%Y-%m-%d"),
            "risk": risks,
            "prob": (probs * 100).round(1).astype(np.float32),
            "fires": test_df["n_fires"].fillna(0).astype(np.int16),
        }))
        log.info("2025 processed in %.2fs (%d rows)", time.time() - t0, len(test_df))

    # ── 2. Process 2023–2024 (Val set: full grid, all 1,737 munis) ───────────────
    if FEATURES_VAL.exists():
        log.info("Processing 2023–2024 validation split...")
        t0 = time.time()
        val_df = pd.read_parquet(
            FEATURES_VAL,
            columns=["cvegeo_inegi", "date", "n_fires"] + feature_cols,
        )
        probs = model.predict_proba(val_df[feature_cols])[:, 1]
        risks = np.zeros(len(probs), dtype=np.int8)
        risks[probs >= med_thr] = 1
        risks[probs >= high_thr] = 2

        dfs.append(pd.DataFrame({
            "cvegeo": val_df["cvegeo_inegi"].astype(str).str.zfill(5),
            "date": pd.to_datetime(val_df["date"]).dt.strftime("%Y-%m-%d"),
            "risk": risks,
            "prob": (probs * 100).round(1).astype(np.float32),
            "fires": val_df["n_fires"].fillna(0).astype(np.int16),
        }))
        log.info("2023–2024 processed in %.2fs (%d rows)", time.time() - t0, len(val_df))

    # ── 3. Process 2016–2022 (Train set) ─────────────────────────────────────────
    if FEATURES_TRAIN.exists():
        log.info("Processing 2016–2022 train split...")
        t0 = time.time()
        train_df = pd.read_parquet(
            FEATURES_TRAIN,
            columns=["cvegeo_inegi", "date", "n_fires"] + feature_cols,
        )
        probs = model.predict_proba(train_df[feature_cols])[:, 1]
        risks = np.zeros(len(probs), dtype=np.int8)
        risks[probs >= med_thr] = 1
        risks[probs >= high_thr] = 2

        dfs.append(pd.DataFrame({
            "cvegeo": train_df["cvegeo_inegi"].astype(str).str.zfill(5),
            "date": pd.to_datetime(train_df["date"]).dt.strftime("%Y-%m-%d"),
            "risk": risks,
            "prob": (probs * 100).round(1).astype(np.float32),
            "fires": train_df["n_fires"].fillna(0).astype(np.int16),
        }))
        log.info("2016–2022 processed in %.2fs (%d rows)", time.time() - t0, len(train_df))

    # ── 4. Process 2015 historical fires & weather ───────────────────────────────
    if FIRES_HARMONIZED.exists():
        log.info("Processing 2015 fire activity...")
        harm = pd.read_parquet(FIRES_HARMONIZED)
        harm_2015 = harm[harm["fecha_inicio"].dt.year == 2015].dropna(subset=["cvegeo_inegi", "fecha_inicio"])
        if not harm_2015.empty:
            fire_days = (
                harm_2015.groupby(["cvegeo_inegi", harm_2015["fecha_inicio"].dt.normalize()])
                .size()
                .reset_index(name="fires")
                .rename(columns={"fecha_inicio": "date"})
            )
            fire_days["cvegeo"] = fire_days["cvegeo_inegi"].astype(str).str.zfill(5)
            fire_days["date"] = pd.to_datetime(fire_days["date"]).dt.strftime("%Y-%m-%d")
            fire_days["risk"] = np.int8(2)  # High risk on fire day
            fire_days["prob"] = np.float32(35.0)
            dfs.append(fire_days[["cvegeo", "date", "risk", "prob", "fires"]])

    # ── 5. Fetch and predict 2026 forecast horizon (today to today + 15) ─────────
    log.info("Fetching 2026 forecast horizon...")
    forecast_df = _build_2026_forecasts(model, feature_cols, med_thr, high_thr)
    if forecast_df is not None and not forecast_df.empty:
        dfs.append(forecast_df)

    log.info("Combining all splits...")
    full_cache = pd.concat(dfs, ignore_index=True)
    # Deduplicate (cvegeo, date), keeping highest risk/prob
    full_cache = (
        full_cache.sort_values(["date", "cvegeo", "prob"], ascending=[True, True, False])
        .drop_duplicates(subset=["date", "cvegeo"])
        .reset_index(drop=True)
    )

    log.info("Saving daily risk cache to %s (%d rows)...", CACHE_FILE, len(full_cache))
    full_cache.to_parquet(CACHE_FILE, index=False, compression="zstd")
    log.info("Cache saved successfully.")
    return full_cache


def _build_2026_forecasts(
    model,
    feature_cols: list[str],
    med_thr: float,
    high_thr: float,
) -> pd.DataFrame | None:
    """Fetch 16-day Open-Meteo forecast across Mexico and predict risk."""
    if not INEGI_MUNICIPALITIES.exists():
        return None

    cat = pd.read_csv(INEGI_MUNICIPALITIES, encoding="utf-8")
    cat["cvegeo"] = cat["cvegeo_inegi"].astype(str).str.zfill(5)

    # 32 state representative centroids
    state_centroids = (
        cat.groupby("estado")[["lat_centroid", "lon_centroid"]]
        .mean()
        .reset_index()
    )

    lats = ",".join(state_centroids["lat_centroid"].round(4).astype(str))
    lons = ",".join(state_centroids["lon_centroid"].round(4).astype(str))
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lats}&longitude={lons}&"
        f"daily=temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,"
        f"precipitation_sum,windspeed_10m_max,windgusts_10m_max,et0_fao_evapotranspiration&"
        f"forecast_days=16&timezone=America/Mexico_City"
    )

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log.warning("Could not fetch 2026 forecast batch: %s", e)
        return None

    if not isinstance(data, list):
        data = [data]

    # Map state -> forecast daily df
    state_forecasts: dict[str, pd.DataFrame] = {}
    for idx, row in state_centroids.iterrows():
        st_name = row["estado"]
        if idx >= len(data):
            break
        d = data[idx].get("daily", {})
        times = d.get("time", [])
        if not times:
            continue
        sdf = pd.DataFrame({
            "date": times,
            "temperature_2m_max": d.get("temperature_2m_max", [np.nan] * len(times)),
            "temperature_2m_min": d.get("temperature_2m_min", [np.nan] * len(times)),
            "relative_humidity_2m_mean": d.get("relative_humidity_2m_mean", [np.nan] * len(times)),
            "precipitation_sum": d.get("precipitation_sum", [np.nan] * len(times)),
            "windspeed_10m_max": d.get("windspeed_10m_max", [np.nan] * len(times)),
            "windgusts_10m_max": d.get("windgusts_10m_max", [np.nan] * len(times)),
            "et0_fao_evapotranspiration": d.get("et0_fao_evapotranspiration", [np.nan] * len(times)),
        })
        state_forecasts[st_name] = sdf

    if not state_forecasts:
        return None

    # Expand to all catalog municipalities
    rows = []
    for _, muni_row in cat.iterrows():
        cve = muni_row["cvegeo"]
        st = muni_row["estado"]
        sdf = state_forecasts.get(st)
        if sdf is None:
            continue
        muni_df = sdf.copy()
        muni_df["cvegeo"] = cve
        rows.append(muni_df)

    if not rows:
        return None

    forecast_all = pd.concat(rows, ignore_index=True)
    forecast_all["month"] = pd.to_datetime(forecast_all["date"]).dt.month
    forecast_all["day_of_year"] = pd.to_datetime(forecast_all["date"]).dt.dayofyear
    forecast_all["sin_day_of_year"] = np.sin(2 * np.pi * forecast_all["day_of_year"] / 365.25)
    forecast_all["cos_day_of_year"] = np.cos(2 * np.pi * forecast_all["day_of_year"] / 365.25)

    # Align columns for model
    pred_X = pd.DataFrame(np.nan, index=forecast_all.index, columns=feature_cols)
    for c in forecast_all.columns:
        if c in pred_X.columns:
            pred_X[c] = forecast_all[c]

    probs = model.predict_proba(pred_X)[:, 1]
    risks = np.zeros(len(probs), dtype=np.int8)
    risks[probs >= med_thr] = 1
    risks[probs >= high_thr] = 2

    return pd.DataFrame({
        "cvegeo": forecast_all["cvegeo"],
        "date": forecast_all["date"],
        "risk": risks,
        "prob": (probs * 100).round(1).astype(np.float32),
        "fires": np.int16(0),
    })


if __name__ == "__main__":
    build_daily_risk_cache()
