"""
tests/test_map_api.py — Verify the new Choropleth Map and Timeline Slider APIs.
"""

import json
import pytest
from app import app, DATE_RISK_CACHE, CVEGEO_INDEX


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_config_endpoint(client):
    """Ensure /api/config returns valid min/max dates for the timeline slider."""
    resp = client.get("/api/config")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "min_date" in data
    assert "max_date" in data
    assert data["min_date"] == "2015-01-01"
    assert data["max_date"] >= "2026-09-18"
    assert "thresholds" in data
    assert data["thresholds"]["medium_threshold"] > 0
    assert data["thresholds"]["high_threshold"] > data["thresholds"]["medium_threshold"]


def test_map_risk_endpoint_historical(client):
    """Ensure /api/map-risk returns instant risk data for a historical peak date."""
    resp = client.get("/api/map-risk?date=2024-05-15")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["date"] == "2024-05-15"
    assert "summary" in data
    assert "risks" in data
    assert len(data["risks"]) > 1000
    assert "01003" in data["risks"]  # Calvillo
    assert data["risks"]["01003"]["risk"] in ["LOW", "MEDIUM", "HIGH"]


def test_map_risk_endpoint_2026(client):
    """Ensure /api/map-risk returns forecast risk data for a 2026 forecast date."""
    resp = client.get("/api/map-risk?date=2026-09-10")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["date"] == "2026-09-10"
    assert "risks" in data
    assert len(data["risks"]) > 1000


def test_municipality_detail_endpoint(client):
    """Ensure /api/municipality-detail returns complete meteo and risk info."""
    resp = client.get("/api/municipality-detail?cvegeo=01003&date=2024-05-15")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["cvegeo"] == "01003"
    assert data["municipio_display"] == "Calvillo"
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert "weather" in data
    assert data["weather"]["temp_max_c"] is not None
    assert data["weather"]["relative_humidity_pct"] is not None


def test_topojson_file_accessible(client):
    """Ensure the static TopoJSON boundary file is served correctly."""
    resp = client.get("/static/data/mexico_municipalities.topojson")
    assert resp.status_code == 200
    assert len(resp.data) > 500000
