"""
src/data/weather.py — Download daily weather data for municipality centroids.

Uses Open-Meteo API (ERA5 reanalysis) with automatic fallback to NASA POWER API
if Open-Meteo rate limits (429) are encountered.

To optimize efficiency and respect API limits, municipality coordinates are
rounded to 0.25° grid resolution (~25km), reducing unique locations from ~1,737
to ~900 unique grid cells while matching ERA5's native spatial resolution.

Usage:
    python -m src.data.weather

Outputs:
    data/external/inegi_municipalities.csv
    data/interim/weather_daily.parquet
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

from src.config import (
    EXTERNAL_DIR,
    INEGI_MUNICIPALITIES,
    OPENMETEO_URL,
    OPENMETEO_FORECAST_URL,
    OPENMETEO_VARIABLES,
    WEATHER_DIR,
    WEATHER_DAILY,
    WEATHER_FETCH_START,
    WEATHER_FETCH_END,
    FIRES_HARMONIZED,
    ERA5_LAG_DAYS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# ── INEGI Municipality Catalog ────────────────────────────────────────────────

def build_municipality_catalog(harmonized: pd.DataFrame) -> pd.DataFrame:
    """Derive municipality catalog from harmonized fire events.

    Uses the mean centroid of all fire event coordinates per CVEGEO as a
    proxy for the municipality centroid.
    """
    log.info("Building municipality catalog from harmonized fire data...")

    catalog = (
        harmonized.dropna(subset=["cvegeo_inegi", "latitud", "longitud"])
        .groupby("cvegeo_inegi")
        .agg(
            cve_ent=("cve_ent", "first"),
            estado=("estado", lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA),
            municipio=("municipio", lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA),
            lat_centroid=("latitud", "mean"),
            lon_centroid=("longitud", "mean"),
            n_fires=("fecha_inicio", "count"),
        )
        .reset_index()
    )

    # Grid cell mapping (0.25° resolution)
    catalog["grid_lat"] = (catalog["lat_centroid"] * 4).round() / 4.0
    catalog["grid_lon"] = (catalog["lon_centroid"] * 4).round() / 4.0

    log.info("Catalog: %d unique municipalities across %d 0.25° grid cells",
             len(catalog), catalog.groupby(["grid_lat", "grid_lon"]).ngroups)
    return catalog


def save_municipality_catalog(catalog: pd.DataFrame) -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(INEGI_MUNICIPALITIES, index=False)
    log.info("Municipality catalog saved to %s", INEGI_MUNICIPALITIES)


def load_municipality_catalog() -> pd.DataFrame:
    if not INEGI_MUNICIPALITIES.exists():
        raise FileNotFoundError(
            f"Municipality catalog not found at {INEGI_MUNICIPALITIES}. "
            "Run src.data.weather first to build it."
        )
    return pd.read_csv(INEGI_MUNICIPALITIES, dtype={"cvegeo_inegi": str})


# ── Weather Fetchers ──────────────────────────────────────────────────────────

def _fetch_openmeteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Try fetching from Open-Meteo API."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(OPENMETEO_VARIABLES),
        "timezone": "America/Mexico_City",
    }
    try:
        resp = requests.get(OPENMETEO_URL, params=params, timeout=15)
        if resp.status_code != 200:
            return None
        data = resp.json()
        daily = data.get("daily", {})
        if not daily or "time" not in daily:
            return None
        df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
        for var in OPENMETEO_VARIABLES:
            df[var] = daily.get(var, [None] * len(df))
        return df
    except Exception:
        return None


def _fetch_nasa_power(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Fallback: Fetch daily weather from NASA POWER API."""
    # NASA date format: YYYYMMDD
    s_date = start.replace("-", "")
    e_date = end.replace("-", "")

    params = {
        "parameters": "T2M_MAX,T2M_MIN,T2M,PRECTOTCORR,WS10M,RH2M",
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": s_date,
        "end": e_date,
        "format": "JSON",
    }
    try:
        resp = requests.get(NASA_POWER_URL, params=params, timeout=30)
        if resp.status_code != 200:
            return None
        data = resp.json()
        params_data = data.get("properties", {}).get("parameter", {})

        dates_str = list(params_data.get("T2M", {}).keys())
        if not dates_str:
            return None

        dates = pd.to_datetime(dates_str, format="%Y%m%d")

        tmax = np.array([params_data.get("T2M_MAX", {}).get(d, np.nan) for d in dates_str])
        tmin = np.array([params_data.get("T2M_MIN", {}).get(d, np.nan) for d in dates_str])
        tmean = np.array([params_data.get("T2M", {}).get(d, np.nan) for d in dates_str])
        precip = np.array([params_data.get("PRECTOTCORR", {}).get(d, np.nan) for d in dates_str])
        wind = np.array([params_data.get("WS10M", {}).get(d, np.nan) for d in dates_str])
        rh = np.array([params_data.get("RH2M", {}).get(d, np.nan) for d in dates_str])

        # Replace NASA error sentinel (-999) with NaN
        for arr in [tmax, tmin, tmean, precip, wind, rh]:
            arr[arr == -999.0] = np.nan

        df = pd.DataFrame({
            "date": dates,
            "temperature_2m_max": tmax,
            "temperature_2m_min": tmin,
            "temperature_2m_mean": tmean,
            "precipitation_sum": precip,
            "rain_sum": precip,
            "windspeed_10m_max": wind,
            "windgusts_10m_max": wind * 1.3,
            "et0_fao_evapotranspiration": np.nan,  # derived if needed
            "shortwave_radiation_sum": np.nan,
            "relative_humidity_2m_mean": rh,
        })
        return df
    except Exception as e:
        log.warning("NASA POWER fetch failed for (%f, %f): %s", lat, lon, e)
        return None


def fetch_grid_cell_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame | None:
    """Fetch daily weather for a grid cell using Open-Meteo with NASA POWER fallback."""
    # Try Open-Meteo first
    df = _fetch_openmeteo(lat, lon, start, end)
    if df is not None:
        return df

    # Fallback to NASA POWER
    df = _fetch_nasa_power(lat, lon, start, end)
    return df


def _fetch_openmeteo_forecast(
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Fetch weather from Open-Meteo Forecast API.

    Covers recent past (~5 days back) through up to 16 days ahead.
    Returns same variable schema as _fetch_openmeteo() / ERA5 archive.

    Parameters
    ----------
    lat, lon : float
        Municipality centroid coordinates.
    start, end : str
        Date range strings in YYYY-MM-DD format.
    """
    from datetime import date as _date
    today = _date.today()
    # past_days: how far back from today to include
    start_dt = pd.Timestamp(start).date()
    past_days = max(0, (today - start_dt).days)
    # forecast_days: how far ahead from today to include
    end_dt = pd.Timestamp(end).date()
    forecast_days = max(1, (end_dt - today).days + 1)

    params = {
        "latitude":      lat,
        "longitude":     lon,
        "daily":         ",".join(OPENMETEO_VARIABLES),
        "timezone":      "America/Mexico_City",
        "past_days":     min(past_days, 92),   # API max is 92
        "forecast_days": min(forecast_days, 16),  # API max is 16
    }
    try:
        resp = requests.get(OPENMETEO_FORECAST_URL, params=params, timeout=15)
        if resp.status_code != 200:
            log.warning(
                "Forecast API returned %d for (%.4f, %.4f)", resp.status_code, lat, lon
            )
            return None
        data = resp.json()
        daily = data.get("daily", {})
        if not daily or "time" not in daily:
            return None
        df = pd.DataFrame({"date": pd.to_datetime(daily["time"])})
        for var in OPENMETEO_VARIABLES:
            df[var] = daily.get(var, [None] * len(df))
        # Trim to requested window
        df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]
        return df if not df.empty else None
    except Exception as exc:
        log.warning("Forecast API fetch failed for (%.4f, %.4f): %s", lat, lon, exc)
        return None


def fetch_weather_window(
    lat: float,
    lon: float,
    start: str,
    end: str,
) -> pd.DataFrame | None:
    """Smart weather fetcher: routes to ERA5 archive vs forecast API based on dates.

    Strategy
    --------
    ERA5 archive has confirmed data up to ~ERA5_LAG_DAYS days ago.
    The forecast API covers recent past (~5 days) + up to 16 days ahead.

    - Dates entirely in the past  → ERA5 archive only
    - Dates entirely near/future  → Forecast API only
    - Mixed window                → Fetch both, concatenate, deduplicate

    Falls back to NASA POWER if both Open-Meteo APIs fail.

    Parameters
    ----------
    lat, lon : float
        Centroid coordinates for the municipality.
    start, end : str
        Date range strings in YYYY-MM-DD format (inclusive).

    Returns
    -------
    pd.DataFrame | None
        DataFrame with columns: date + all OPENMETEO_VARIABLES columns.
    """
    from datetime import date as _date, timedelta

    today = _date.today()
    # ERA5 archive is reliable up to this date
    archive_end_dt  = today - timedelta(days=ERA5_LAG_DAYS)
    archive_end_str = archive_end_dt.strftime("%Y-%m-%d")

    start_dt = pd.Timestamp(start).date()
    end_dt   = pd.Timestamp(end).date()

    frames: list[pd.DataFrame] = []

    # ── ERA5 archive segment ──────────────────────────────────────────────────
    if start_dt <= archive_end_dt:
        archive_seg_end = min(end_dt, archive_end_dt).strftime("%Y-%m-%d")
        df_arch = _fetch_openmeteo(lat, lon, start, archive_seg_end)
        if df_arch is not None:
            frames.append(df_arch)
        else:
            log.debug("ERA5 archive unavailable for (%.4f, %.4f) %s–%s",
                      lat, lon, start, archive_seg_end)

    # ── Forecast API segment ─────────────────────────────────────────────────
    if end_dt > archive_end_dt:
        # Start forecast segment from max(archive_end+1, start)
        forecast_seg_start_dt = max(start_dt, archive_end_dt + timedelta(days=1))
        forecast_seg_start = forecast_seg_start_dt.strftime("%Y-%m-%d")
        df_fcst = _fetch_openmeteo_forecast(lat, lon, forecast_seg_start, end)
        if df_fcst is not None:
            frames.append(df_fcst)
        else:
            log.debug("Forecast API unavailable for (%.4f, %.4f) %s–%s",
                      lat, lon, forecast_seg_start, end)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["date"] = pd.to_datetime(combined["date"])
        combined = (
            combined
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )
        return combined

    # ── Final fallback: NASA POWER ────────────────────────────────────────────
    log.warning(
        "Both Open-Meteo APIs failed for (%.4f, %.4f) — trying NASA POWER.", lat, lon
    )
    return _fetch_nasa_power(lat, lon, start, end)


# ── Bulk Download ─────────────────────────────────────────────────────────────

def download_weather(
    catalog: pd.DataFrame,
    start: str = WEATHER_FETCH_START,
    end: str = WEATHER_FETCH_END,
    resume: bool = True,
) -> pd.DataFrame:
    """Download weather for all grid cells covering the municipality catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Municipality catalog.
    start : str
        Start date.
    end : str
        End date.
    resume : bool
        If True, skip grid cells already cached on disk.

    Returns
    -------
    pd.DataFrame
        Combined weather DataFrame per municipality.
    """
    WEATHER_DIR.mkdir(parents=True, exist_ok=True)

    # Find unique grid cells
    if "grid_lat" not in catalog.columns:
        catalog["grid_lat"] = (catalog["lat_centroid"] * 4).round() / 4.0
        catalog["grid_lon"] = (catalog["lon_centroid"] * 4).round() / 4.0

    grid_cells = (
        catalog.groupby(["grid_lat", "grid_lon"])
        .size()
        .reset_index(name="count")
    )
    log.info("Downloading weather for %d unique 0.25° grid cells...", len(grid_cells))

    grid_cache: dict[tuple[float, float], pd.DataFrame] = {}

    for _, row in tqdm(grid_cells.iterrows(), total=len(grid_cells), desc="Fetching grid cells"):
        glat = round(float(row["grid_lat"]), 4)
        glon = round(float(row["grid_lon"]), 4)
        key = (glat, glon)

        cell_file = WEATHER_DIR / f"grid_{glat:.2f}_{glon:.2f}.parquet"

        if resume and cell_file.exists():
            df_cell = pd.read_parquet(cell_file)
            grid_cache[key] = df_cell
            continue

        df_cell = fetch_grid_cell_weather(glat, glon, start, end)
        if df_cell is not None:
            df_cell.to_parquet(cell_file, index=False)
            grid_cache[key] = df_cell
        else:
            log.warning("Could not fetch weather for grid cell (%f, %f)", glat, glon)

        time.sleep(0.15)  # gentle rate limiting

    log.info("Cached %d/%d grid cells successfully", len(grid_cache), len(grid_cells))

    # Map grid cell weather back to each municipality in the catalog
    all_muni_frames: list[pd.DataFrame] = []
    for _, row in catalog.iterrows():
        cvegeo = str(row["cvegeo_inegi"])
        glat = round(float(row["grid_lat"]), 4)
        glon = round(float(row["grid_lon"]), 4)
        key = (glat, glon)

        df_cell = grid_cache.get(key)
        if df_cell is not None:
            df_muni = df_cell.copy()
            df_muni.insert(0, "cvegeo_inegi", cvegeo)
            all_muni_frames.append(df_muni)

    if not all_muni_frames:
        raise RuntimeError("No weather data was acquired.")

    combined = pd.concat(all_muni_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])

    log.info(
        "Combined weather: %s rows | %d municipalities | %s to %s",
        f"{len(combined):,}",
        combined["cvegeo_inegi"].nunique(),
        combined["date"].min().date(),
        combined["date"].max().date(),
    )

    return combined


def save_weather(weather: pd.DataFrame) -> None:
    WEATHER_DAILY.parent.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(WEATHER_DAILY, index=False)
    log.info("Weather data saved to %s", WEATHER_DAILY)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if INEGI_MUNICIPALITIES.exists():
        log.info("Loading existing municipality catalog...")
        catalog = load_municipality_catalog()
    else:
        if not FIRES_HARMONIZED.exists():
            raise FileNotFoundError(
                "fires_harmonized.parquet not found. Run src.data.harmonize first."
            )
        harmonized = pd.read_parquet(FIRES_HARMONIZED)
        catalog = build_municipality_catalog(harmonized)
        save_municipality_catalog(catalog)

    weather = download_weather(catalog)
    save_weather(weather)


if __name__ == "__main__":
    main()
