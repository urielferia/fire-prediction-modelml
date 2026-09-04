"""
app.py — Flask backend for the Interactive Mexico Wildfire Risk Map.

Routes:
    GET  /                                  → serves static/index.html
    GET  /api/config                        → runtime dates, forecast horizon, & thresholds
    GET  /api/states                        → list of 32 Mexican states
    GET  /api/municipalities?state=<name>   → municipalities for a state
    GET  /api/map-risk?date=<YYYY-MM-DD>    → nationwide daily risk choropleth dictionary (<5ms)
    GET  /api/municipality-detail           → detailed meteorological & risk data for clicked municipality
    POST /api/predict                       → on-demand single prediction

Usage:
    python app.py                  (development mode)
    python app.py --port 8080      (custom port)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.config import (
    INEGI_MUNICIPALITIES,
    MAX_FORECAST_DAYS,
    PROCESSED_DIR,
    WEATHER_DAILY,
    THRESHOLDS,
)

# ── Correct display names for Mexico's 32 states ─────────────────────────────
_STATE_DISPLAY: dict[str, str] = {
    "aguascalientes":      "Aguascalientes",
    "baja california":     "Baja California",
    "baja california sur": "Baja California Sur",
    "campeche":            "Campeche",
    "chiapas":             "Chiapas",
    "chihuahua":           "Chihuahua",
    "coahuila":            "Coahuila",
    "colima":              "Colima",
    "durango":             "Durango",
    "guanajuato":          "Guanajuato",
    "guerrero":            "Guerrero",
    "hidalgo":             "Hidalgo",
    "jalisco":             "Jalisco",
    "morelos":             "Morelos",
    "nayarit":             "Nayarit",
    "oaxaca":              "Oaxaca",
    "puebla":              "Puebla",
    "quintana roo":        "Quintana Roo",
    "sinaloa":             "Sinaloa",
    "sonora":              "Sonora",
    "tabasco":             "Tabasco",
    "tamaulipas":          "Tamaulipas",
    "tlaxcala":            "Tlaxcala",
    "veracruz":            "Veracruz",
    "zacatecas":           "Zacatecas",
    "ciudad de ma\ufffdxico": "Mexico City",
    "ma\ufffdxico":           "State of Mexico",
    "michoaca\ufffdn":        "Michoacan",
    "nuevo lea\ufffdn":       "Nuevo Leon",
    "quera\ufffftaro":        "Queretaro",
    "queractaro":           "Queretaro",
    "san luis potosa\ufffd": "San Luis Potosi",
    "yucata\ufffdn":          "Yucatan",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
CACHE_FILE = PROCESSED_DIR / "daily_risk_cache.parquet"

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
CORS(app)


# ── Preload INEGI catalog once at startup ────────────────────────────────────

def _load_catalog() -> tuple[pd.DataFrame, dict[str, dict]]:
    """Load and index the INEGI municipalities catalog."""
    df = pd.read_csv(INEGI_MUNICIPALITIES, encoding="utf-8")
    df["cvegeo_inegi"] = df["cvegeo_inegi"].astype(str).str.zfill(5)
    df["estado_norm"] = df["estado"].str.strip().str.lower()
    df["municipio_norm"] = df["municipio"].str.strip().str.lower()
    df["estado_display"] = df["estado_norm"].map(
        lambda v: _STATE_DISPLAY.get(v, v.title())
    )
    df["municipio_display"] = df["municipio"].str.title()

    cve_idx = {}
    for _, row in df.iterrows():
        cve_idx[row["cvegeo_inegi"]] = {
            "cvegeo": row["cvegeo_inegi"],
            "municipio": row["municipio"],
            "estado": row["estado"],
            "municipio_display": row["municipio_display"],
            "estado_display": row["estado_display"],
            "lat": float(row["lat_centroid"]),
            "lon": float(row["lon_centroid"]),
        }
    return df, cve_idx


# ── Preload daily risk cache for instant map scrubbing ──────────────────────

RISK_CODES = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}

def _load_risk_cache() -> dict[str, dict]:
    """Load precomputed risk cache into date-indexed memory structure."""
    if not os.path.exists(CACHE_FILE):
        log.warning("Daily risk cache not found at %s. Run src.features.risk_cache first.", CACHE_FILE)
        return {}

    t0 = time.time()
    df = pd.read_parquet(CACHE_FILE)
    cache: dict[str, dict] = {}

    for dt, grp in df.groupby("date"):
        cves = grp["cvegeo"].tolist()
        risks = grp["risk"].tolist()
        probs = grp["prob"].tolist()
        fires = grp["fires"].tolist()

        muni_map = {}
        high_cnt = med_cnt = low_cnt = fire_cnt = 0

        for c, r, p, fi in zip(cves, risks, probs, fires):
            r_str = RISK_CODES.get(r, "LOW")
            if r == 2:
                high_cnt += 1
            elif r == 1:
                med_cnt += 1
            else:
                low_cnt += 1
            if fi > 0:
                fire_cnt += int(fi)

            muni_map[c] = {
                "risk": r_str,
                "prob": round(float(p), 1),
                "fires": int(fi),
            }

        cache[dt] = {
            "summary": {
                "high": high_cnt,
                "medium": med_cnt,
                "low": low_cnt,
                "fires": fire_cnt,
            },
            "risks": muni_map,
        }

    log.info("Loaded daily risk cache in %.2fs: %d dates indexed.", time.time() - t0, len(cache))
    return cache


import time
CATALOG, CVEGEO_INDEX = _load_catalog()
DATE_RISK_CACHE = _load_risk_cache()


def _get_thresholds() -> dict:
    try:
        with open(THRESHOLDS) as f:
            thr = json.load(f)
        return {
            "medium_threshold": round(thr["medium_threshold"] * 100, 2),
            "high_threshold":   round(thr["high_threshold"] * 100, 2),
        }
    except Exception:
        return {"medium_threshold": 2.12, "high_threshold": 19.27}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the main frontend page."""
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/config")
def get_config():
    """Return runtime configuration for timeline slider and legends."""
    today = date.today()
    max_date = today + timedelta(days=MAX_FORECAST_DAYS)
    return jsonify({
        "min_date": "2015-01-01",
        "max_date": max_date.isoformat(),
        "today": today.isoformat(),
        "default_date": "2024-05-15",
        "max_forecast_days": MAX_FORECAST_DAYS,
        "thresholds": _get_thresholds(),
        "total_municipalities": len(CVEGEO_INDEX),
        "available_dates_count": len(DATE_RISK_CACHE),
    })


@app.route("/api/states")
def get_states():
    """Return list of 32 Mexican states, sorted alphabetically."""
    if CATALOG.empty:
        return jsonify({"error": "Catalog not available"}), 500

    states = (
        CATALOG[["estado_norm", "estado_display"]]
        .drop_duplicates("estado_norm")
        .sort_values("estado_display")
        .rename(columns={"estado_norm": "value", "estado_display": "label"})
        .to_dict(orient="records")
    )
    return jsonify(states)


@app.route("/api/municipalities")
def get_municipalities():
    """Return municipalities for the given state."""
    if CATALOG.empty:
        return jsonify({"error": "Catalog not available"}), 500

    state = request.args.get("state", "").strip().lower()
    if not state:
        return jsonify({"error": "Missing 'state' query parameter"}), 400

    subset = CATALOG[CATALOG["estado_norm"] == state].copy()
    if subset.empty:
        return jsonify({"error": f"State '{state}' not found"}), 404

    munis = (
        subset[["municipio_norm", "municipio_display", "lat_centroid", "lon_centroid", "cvegeo_inegi"]]
        .sort_values("municipio_display")
        .rename(columns={
            "municipio_norm": "value",
            "municipio_display": "label",
            "lat_centroid": "lat",
            "lon_centroid": "lon",
        })
        .to_dict(orient="records")
    )
    return jsonify(munis)


@app.route("/api/map-risk")
def get_map_risk():
    """Return fast nationwide choropleth risk dictionary for the given date.

    Query params:
        date (str): YYYY-MM-DD
    """
    target_date = request.args.get("date", "").strip()
    if not target_date:
        return jsonify({"error": "Missing 'date' parameter (YYYY-MM-DD)"}), 400

    # Look up in precomputed in-memory cache
    if target_date in DATE_RISK_CACHE:
        entry = DATE_RISK_CACHE[target_date]
        return jsonify({
            "date": target_date,
            "found": True,
            "summary": entry["summary"],
            "risks": entry["risks"],
        })

    # If date is not in precomputed cache, return default baseline low risk
    return jsonify({
        "date": target_date,
        "found": False,
        "summary": {"high": 0, "medium": 0, "low": len(CVEGEO_INDEX), "fires": 0},
        "risks": {},
    })


@app.route("/api/municipality-detail")
def get_municipality_detail():
    """Return detailed meteorological conditions, risk level, and historical fire info

    Query params:
        cvegeo (str): 5-digit CVEGEO (e.g. '01003')
        date (str): YYYY-MM-DD
    """
    cvegeo = request.args.get("cvegeo", "").strip().zfill(5)
    fecha = request.args.get("date", "").strip()

    if not cvegeo or not fecha:
        return jsonify({"error": "cvegeo and date parameters are required"}), 400

    muni_meta = CVEGEO_INDEX.get(cvegeo)
    if not muni_meta:
        return jsonify({"error": f"CVEGEO '{cvegeo}' not found in catalog"}), 404

    thresholds = _get_thresholds()

    # Determine risk from date cache if present
    day_entry = DATE_RISK_CACHE.get(fecha, {}).get("risks", {}).get(cvegeo)
    if day_entry:
        risk_level = day_entry["risk"]
        probability = day_entry["prob"]
        fires_count = day_entry["fires"]
    else:
        risk_level = "LOW"
        probability = 1.0
        fires_count = 0

    # Parse date to check data source
    try:
        dt_obj = datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": f"Invalid date format: '{fecha}'"}), 400

    weather_data: dict = {}
    data_source = "ERA5 Reanalysis (Historical)"

    # If date is in the historical parquet era (<= 2025-12-31), read actual measurements
    if dt_obj.year <= 2025 and WEATHER_DAILY.exists():
        try:
            ts = pd.Timestamp(fecha)
            w_row = pd.read_parquet(
                WEATHER_DAILY,
                filters=[("cvegeo_inegi", "==", cvegeo), ("date", "==", ts)],
            )
            if not w_row.empty:
                r0 = w_row.iloc[0]
                def _val(col):
                    v = r0.get(col)
                    return None if pd.isna(v) else round(float(v), 2)

                weather_data = {
                    "temp_max_c": _val("temperature_2m_max"),
                    "temp_min_c": _val("temperature_2m_min"),
                    "temp_mean_c": _val("temperature_2m_mean"),
                    "relative_humidity_pct": _val("relative_humidity_2m_mean"),
                    "precipitation_mm": _val("precipitation_sum"),
                    "wind_speed_kmh": _val("windspeed_10m_max"),
                    "wind_gusts_kmh": _val("windgusts_10m_max"),
                    "evapotranspiration_mm": _val("et0_fao_evapotranspiration"),
                    "consecutive_dry_days": 0 if (_val("precipitation_sum") or 0) > 0.5 else 5,
                }
        except Exception as e:
            log.warning("Could not read weather_daily for %s on %s: %s", cvegeo, fecha, e)

    # For 2026 or missing historical weather, fall back to predict_risk
    if not weather_data:
        from src.predict import predict_risk
        pred_res = predict_risk(muni_meta["municipio"], muni_meta["estado"], fecha, verbose=False)
        if pred_res.get("weather"):
            weather_data = pred_res["weather"]
        if pred_res.get("probability") is not None:
            probability = pred_res["probability"]
            risk_level = pred_res["risk_level"]
        data_source = pred_res.get("data_source", "Open-Meteo Forecast")

    return jsonify({
        "cvegeo": cvegeo,
        "municipio": muni_meta["municipio"],
        "estado": muni_meta["estado"],
        "municipio_display": muni_meta["municipio_display"],
        "estado_display": muni_meta["estado_display"],
        "lat": muni_meta["lat"],
        "lon": muni_meta["lon"],
        "date": fecha,
        "risk_level": risk_level,
        "probability": probability,
        "fires_count": fires_count,
        "weather": weather_data,
        "thresholds": thresholds,
        "data_source": data_source,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Run full wildfire risk prediction pipeline on-demand."""
    body = request.get_json(force=True, silent=True) or {}

    municipio = body.get("municipio", "").strip()
    estado = body.get("estado", "").strip()
    fecha = body.get("fecha", "").strip()

    if not municipio or not estado or not fecha:
        return jsonify({"error": "Municipality, state, and date are required"}), 400

    from src.predict import predict_risk
    result = predict_risk(municipio, estado, fecha, verbose=False)

    # Enrich with lat/lon from catalog
    if result.get("found_municipality") and not CATALOG.empty:
        from src.data.harmonize import _normalize_text
        mask = (
            (CATALOG["estado_norm"] == _normalize_text(estado))
            & (CATALOG["municipio_norm"] == _normalize_text(municipio))
        )
        row = CATALOG[mask]
        if not row.empty:
            result["lat"] = float(row.iloc[0]["lat_centroid"])
            result["lon"] = float(row.iloc[0]["lon_centroid"])

    result["thresholds"] = _get_thresholds()
    return jsonify(result)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire Risk Map — Flask server")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    log.info("Starting Wildfire Risk Map on http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
