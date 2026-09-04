"""
src/predict.py — Wildfire risk prediction interface.

Accepts a municipality name + state + date (+ optional weather conditions)
and returns a probability estimate with risk level classification.

Usage (CLI):
    python -m src.predict --municipio "Calvillo" --estado "Aguascalientes" --fecha "2025-05-15"

Usage (Python):
    from src.predict import predict_risk
    result = predict_risk("Calvillo", "Aguascalientes", "2025-05-15")
    print(result)
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime, timedelta

import joblib
import numpy as np
import pandas as pd

from src.config import (
    CALIBRATOR,
    FIRES_HARMONIZED,
    THRESHOLDS,
    WEATHER_DAILY,
    WEATHER_FETCH_START,
    MAX_FORECAST_DAYS,
)
from src.models.calibrate import classify_risk
from src.features.temporal import add_temporal_features
from src.features.weather import add_weather_features, _RAW_WEATHER
from src.features.historical import add_historical_features
from src.features.static import add_static_features, build_static_features
from src.data.weather import fetch_weather_window

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_artifacts():
    """Load model, thresholds, and supporting data once."""
    bundle = joblib.load(CALIBRATOR)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]

    with open(THRESHOLDS) as f:
        thresholds = json.load(f)

    harmonized = pd.read_parquet(FIRES_HARMONIZED)
    weather = pd.read_parquet(WEATHER_DAILY)

    return model, feature_cols, thresholds, harmonized, weather


def _find_municipality(
    municipio: str,
    estado: str,
    harmonized: pd.DataFrame,
) -> dict | None:
    """Look up municipality CVEGEO and centroid from harmonized data."""
    from src.data.harmonize import _normalize_text

    muni_norm = _normalize_text(municipio)
    state_norm = _normalize_text(estado)

    mask = (
        harmonized["municipio"].map(_normalize_text) == muni_norm
    ) & (
        harmonized["estado"].map(_normalize_text) == state_norm
    )

    subset = harmonized[mask]
    if subset.empty:
        # Try partial match on municipality name
        mask2 = harmonized["municipio"].map(
            lambda x: _normalize_text(str(x)).startswith(muni_norm[:5])
        ) & (harmonized["estado"].map(_normalize_text) == state_norm)
        subset = harmonized[mask2]

    if subset.empty:
        return None

    row = subset.iloc[0]
    return {
        "cvegeo_inegi": str(row["cvegeo_inegi"]),
        "municipio": str(row["municipio"]),
        "estado": str(row["estado"]),
        "lat": float(row["latitud"]),
        "lon": float(row["longitud"]),
    }


def _build_prediction_row(
    cvegeo: str,
    target_date: datetime,
    harmonized: pd.DataFrame,
    weather: pd.DataFrame,
    static_features: pd.DataFrame,
    feature_cols: list[str],
    lat: float | None = None,
    lon: float | None = None,
) -> pd.DataFrame:
    """Build a single-row feature DataFrame for a prediction.

    For dates beyond the stored weather_daily.parquet, fetches the missing
    window on-demand using the hybrid ERA5 / forecast API router.
    """
    from datetime import date as _date

    # We need a small window of history for rolling features
    # Use 60 days before the target date
    window_start = target_date - timedelta(days=60)

    # ── Fast path: pull from stored parquet ──────────────────────────────────
    muni_weather = weather[
        (weather["cvegeo_inegi"] == cvegeo)
        & (weather["date"] >= window_start)
        & (weather["date"] <= target_date)
    ].copy()

    # ── On-demand fetch for dates not in the stored parquet ──────────────────
    # Determine what range is missing
    stored_max = (
        pd.Timestamp(muni_weather["date"].max())
        if not muni_weather.empty
        else pd.Timestamp(window_start) - timedelta(days=1)
    )
    fetch_start = stored_max + timedelta(days=1)
    fetch_end   = pd.Timestamp(target_date)

    if fetch_start <= fetch_end and lat is not None and lon is not None:
        log.info(
            "Fetching weather on-demand for %s: %s → %s",
            cvegeo,
            fetch_start.date(),
            fetch_end.date(),
        )
        fresh_weather = fetch_weather_window(
            lat=lat,
            lon=lon,
            start=fetch_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
        )
        if fresh_weather is not None and not fresh_weather.empty:
            fresh_weather.insert(0, "cvegeo_inegi", cvegeo)
            fresh_weather["date"] = pd.to_datetime(fresh_weather["date"])
            muni_weather = pd.concat(
                [muni_weather, fresh_weather], ignore_index=True
            ).drop_duplicates("date")
        else:
            log.warning(
                "Could not fetch weather for %s %s–%s — weather features may be NaN.",
                cvegeo, fetch_start.date(), fetch_end.date(),
            )
    elif fetch_start <= fetch_end:
        log.warning(
            "No lat/lon provided for on-demand fetch — weather features may be NaN for %s.",
            cvegeo,
        )

    # Ensure we have at least a skeleton for the full window
    if muni_weather.empty:
        dates = pd.date_range(window_start, target_date, freq="D")
        muni_weather = pd.DataFrame({"cvegeo_inegi": cvegeo, "date": dates})
        for v in _RAW_WEATHER:
            muni_weather[v] = np.nan

    # Build municipality-day rows for the window
    dates = pd.date_range(window_start, target_date, freq="D")
    rows = pd.DataFrame({"cvegeo_inegi": cvegeo, "date": dates})
    rows["fire_occurred"] = 0

    # Join weather
    muni_weather["date"] = pd.to_datetime(muni_weather["date"])
    rows = rows.merge(muni_weather.drop(columns=["cvegeo_inegi"], errors="ignore"),
                      on="date", how="left")

    # Add temporal features
    rows = add_temporal_features(rows)

    # Add weather rolling features
    rows = add_weather_features(rows)

    # Add historical fire features (using historical data up to target_date)
    hist_subset = harmonized[harmonized["fecha_inicio"] < target_date].copy()
    rows = add_historical_features(rows, hist_subset)

    # Add static features
    rows = add_static_features(rows, static_features)

    # Extract just the target date row
    target_row = rows[rows["date"] == pd.Timestamp(target_date)].copy()

    if target_row.empty:
        log.warning("Could not build feature row for %s on %s", cvegeo, target_date.date())
        return pd.DataFrame(), {}

    # Extract raw and key rolling meteorological values for explainability & UI
    weather_data = {}
    metric_mapping = [
        ("temp_max", "temp_max_c"),
        ("temp_min", "temp_min_c"),
        ("temp_mean", "temp_mean_c"),
        ("humidity", "relative_humidity_pct"),
        ("precip", "precipitation_mm"),
        ("wind_max", "wind_speed_kmh"),
        ("wind_gusts", "wind_gusts_kmh"),
        ("et0", "evapotranspiration_mm"),
        ("precip_sum_30d", "precip_sum_30d_mm"),
        ("temp_max_7d_mean", "temp_max_7d_mean_c"),
        ("dry_streak", "consecutive_dry_days"),
    ]
    for src_col, out_key in metric_mapping:
        if src_col in target_row.columns and not pd.isna(target_row[src_col].iloc[0]):
            weather_data[out_key] = round(float(target_row[src_col].iloc[0]), 2)
        else:
            weather_data[out_key] = None

    # Select model feature columns (fill missing with NaN)
    for col in feature_cols:
        if col not in target_row.columns:
            target_row[col] = np.nan

    return target_row[feature_cols], weather_data


# ── Main prediction function ───────────────────────────────────────────────────

def predict_risk(
    municipio: str,
    estado: str,
    fecha: str,
    verbose: bool = True,
) -> dict:
    """Predict wildfire risk for a municipality on a given date.

    Parameters
    ----------
    municipio : str
        Municipality name (Spanish, e.g. "Calvillo").
    estado : str
        State name (Spanish, e.g. "Aguascalientes").
    fecha : str
        Date string in YYYY-MM-DD format.
    verbose : bool
        If True, print formatted output to console.

    Returns
    -------
    dict
        {
            'municipio': str,
            'estado': str,
            'date': str,
            'probability': float,
            'risk_level': str ('LOW' | 'MEDIUM' | 'HIGH'),
            'found_municipality': bool,
            'warning': str | None,
        }
    """
    result = {
        "municipio": municipio,
        "estado": estado,
        "date": fecha,
        "probability": None,
        "risk_level": None,
        "found_municipality": False,
        "warning": None,
        "data_source": None,
        "weather": None,
    }

    # Load artifacts
    try:
        model, feature_cols, thresholds, harmonized, weather = _load_artifacts()
    except FileNotFoundError as e:
        result["warning"] = f"Model artifacts not found: {e}. Run the full pipeline first."
        if verbose:
            print(result["warning"])
        return result

    # Parse date
    try:
        target_date = datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        result["warning"] = f"Invalid date format: '{fecha}'. Use YYYY-MM-DD."
        if verbose:
            print(result["warning"])
        return result

    # Enforce date ceiling: today + MAX_FORECAST_DAYS
    from datetime import date as _date
    max_allowed = _date.today() + timedelta(days=MAX_FORECAST_DAYS)
    if target_date.date() > max_allowed:
        result["warning"] = (
            f"Date '{fecha}' exceeds the forecast horizon ({max_allowed}). "
            f"Maximum allowed: today + {MAX_FORECAST_DAYS} days."
        )
        if verbose:
            print(result["warning"])
        return result

    # Determine data source label for transparency
    from datetime import timedelta as _td
    era5_cutoff = _date.today() - _td(days=6)
    if target_date.date() <= era5_cutoff:
        result["data_source"] = "ERA5 reanalysis (archive)"
    else:
        result["data_source"] = "Open-Meteo forecast (16-day)"

    # Find municipality
    muni_info = _find_municipality(municipio, estado, harmonized)
    if muni_info is None:
        result["warning"] = (
            f"Municipality '{municipio}' in '{estado}' not found in fire records. "
            "Prediction unavailable for municipalities with no historical fire data."
        )
        if verbose:
            print(result["warning"])
        return result

    result["found_municipality"] = True
    cvegeo = muni_info["cvegeo_inegi"]
    lat    = muni_info["lat"]
    lon    = muni_info["lon"]

    # Build static features
    static_features = build_static_features(harmonized, catalog=None)

    # Build feature row (passes lat/lon for on-demand weather fetch)
    X, weather_data = _build_prediction_row(
        cvegeo, target_date, harmonized, weather, static_features, feature_cols,
        lat=lat, lon=lon,
    )

    if X.empty:
        result["warning"] = "Could not assemble feature vector for this municipality/date."
        if verbose:
            print(result["warning"])
        return result

    result["weather"] = weather_data

    # Predict
    prob = float(model.predict_proba(X)[0, 1])
    risk = classify_risk(prob, thresholds)

    result["probability"] = round(prob * 100, 2)   # expressed as percentage
    result["risk_level"] = risk
    result["cvegeo_inegi"] = cvegeo

    if verbose:
        _print_prediction(result, muni_info)

    return result


def _print_prediction(result: dict, muni_info: dict) -> None:
    risk_tags = {"LOW": "[LOW]", "MEDIUM": "[MEDIUM]", "HIGH": "[HIGH]"}
    tag = risk_tags.get(result["risk_level"], "[UNKNOWN]")

    print()
    print("=" * 50)
    print("  WILDFIRE RISK PREDICTION - MEXICO")
    print("=" * 50)
    print(f"  Municipality : {muni_info['municipio'].title()}")
    print(f"  State        : {muni_info['estado'].title()}")
    print(f"  Date         : {result['date']}")
    print(f"  Lat / Lon    : {muni_info['lat']:.4f}, {muni_info['lon']:.4f}")
    print("-" * 50)
    print(f"  Wildfire probability : {result['probability']:.1f}%")
    print(f"  Risk level           : {tag} {result['risk_level']}")
    if result.get("weather"):
        w = result["weather"]
        print("-" * 50)
        print("  METEOROLOGICAL CONDITIONS:")
        if w.get("temp_max_c") is not None:
            print(f"  Max Temperature      : {w['temp_max_c']} °C")
        if w.get("temp_min_c") is not None:
            print(f"  Min Temperature      : {w['temp_min_c']} °C")
        if w.get("relative_humidity_pct") is not None:
            print(f"  Relative Humidity    : {w['relative_humidity_pct']} %")
        if w.get("precipitation_mm") is not None:
            print(f"  Precipitation (day)  : {w['precipitation_mm']} mm")
        if w.get("wind_speed_kmh") is not None:
            print(f"  Wind Speed (max)     : {w['wind_speed_kmh']} km/h")
        if w.get("wind_gusts_kmh") is not None:
            print(f"  Wind Gusts (max)     : {w['wind_gusts_kmh']} km/h")
        if w.get("precip_sum_30d_mm") is not None:
            print(f"  30-Day Rain Sum      : {w['precip_sum_30d_mm']} mm")
    if result.get("warning"):
        print(f"  Warning              : {result['warning']}")
    print("=" * 50)
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict wildfire risk for a Mexican municipality on a given date."
    )
    parser.add_argument("--municipio", required=True, help="Municipality name in Spanish")
    parser.add_argument("--estado", required=True, help="State name in Spanish")
    parser.add_argument("--fecha", required=True, help="Date in YYYY-MM-DD format")
    args = parser.parse_args()

    predict_risk(args.municipio, args.estado, args.fecha, verbose=True)


if __name__ == "__main__":
    main()
