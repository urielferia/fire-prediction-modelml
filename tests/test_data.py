"""
tests/test_data.py — Unit tests for data loading, harmonization, and feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.load import load_raw_2015_2024, load_raw_2025
from src.data.harmonize import _dms_to_decimal, _normalize_text, _reconstruct_inegi_cvegeo


class TestLoad:
    def test_load_2015_2024_returns_dataframe(self):
        df = load_raw_2015_2024()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "Fecha_Inicio" in df.columns
        assert "CVE_ENT" in df.columns

    def test_load_2015_2024_row_count(self):
        df = load_raw_2015_2024()
        assert 60_000 < len(df) < 80_000

    def test_load_2025_returns_dataframe(self):
        df = load_raw_2025()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "fecha_inicio" in df.columns

    def test_load_2025_row_count(self):
        df = load_raw_2025()
        assert 5_000 < len(df) < 10_000


class TestHarmonize:
    def test_normalize_text_lowercase(self):
        assert _normalize_text("Intencional") == "intencional"

    def test_normalize_text_strips_whitespace(self):
        assert _normalize_text("  fogatas  ") == "fogatas"

    def test_normalize_text_removes_accents(self):
        result = _normalize_text("Michoacán")
        assert "á" not in result
        assert "michoacan" == result

    def test_dms_to_decimal_known_value(self):
        lat = _dms_to_decimal(
            pd.Series([21]), pd.Series([53]), pd.Series([11.6])
        )
        assert abs(lat.iloc[0] - 21.8866) < 0.001

    def test_dms_to_decimal_zero(self):
        result = _dms_to_decimal(pd.Series([0]), pd.Series([0]), pd.Series([0.0]))
        assert result.iloc[0] == 0.0

    def test_reconstruct_inegi_cvegeo_format(self):
        ent = pd.Series([1])
        mun = pd.Series([3.0])
        result = _reconstruct_inegi_cvegeo(ent, mun)
        assert result.iloc[0] == "01003"

    def test_reconstruct_inegi_cvegeo_cdmx(self):
        ent = pd.Series([9])
        mun = pd.Series([13.0])
        result = _reconstruct_inegi_cvegeo(ent, mun)
        assert result.iloc[0] == "09013"


class TestTemporalFeatures:
    def test_add_temporal_features(self):
        from src.features.temporal import add_temporal_features
        df = pd.DataFrame({"date": pd.date_range("2024-03-15", periods=5, freq="D")})
        result = add_temporal_features(df)
        assert "month" in result.columns
        assert "day_of_year" in result.columns
        assert "is_fire_season" in result.columns
        assert "sin_day_of_year" in result.columns
        assert result["is_fire_season"].iloc[0] == 1

    def test_season_assignment(self):
        from src.features.temporal import add_temporal_features
        df = pd.DataFrame({"date": ["2024-01-15", "2024-04-15", "2024-08-15"]})
        result = add_temporal_features(df)
        assert result["season"].iloc[0] == "dry_cool"
        assert result["season"].iloc[1] == "dry_hot"
        assert result["season"].iloc[2] == "wet"

    def test_cyclical_encoding_range(self):
        from src.features.temporal import add_temporal_features
        df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=366, freq="D")})
        result = add_temporal_features(df)
        assert result["sin_day_of_year"].between(-1, 1).all()
        assert result["cos_day_of_year"].between(-1, 1).all()


class TestWeatherFeatures:
    def test_add_weather_features_lags(self):
        from src.features.weather import add_weather_features
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        df = pd.DataFrame({
            "cvegeo_inegi": "01001",
            "date": dates,
            "temperature_2m_max": np.linspace(20, 29, 10),
            "temperature_2m_min": np.linspace(5, 14, 10),
            "temperature_2m_mean": np.linspace(12, 21, 10),
            "precipitation_sum": [0, 0, 5, 0, 0, 0, 0, 10, 0, 0],
            "rain_sum": [0, 0, 5, 0, 0, 0, 0, 10, 0, 0],
            "windspeed_10m_max": np.ones(10) * 15,
            "windgusts_10m_max": np.ones(10) * 25,
            "et0_fao_evapotranspiration": np.ones(10) * 3,
            "shortwave_radiation_sum": np.ones(10) * 20,
            "relative_humidity_2m_mean": np.ones(10) * 40,
        })
        feat = add_weather_features(df)
        assert "temp_max_lag1" in feat.columns
        assert "precip_sum_7d" in feat.columns
        assert "dry_streak" in feat.columns
        # Check lag1
        assert pd.isna(feat["temp_max_lag1"].iloc[0])
        assert feat["temp_max_lag1"].iloc[1] == 20.0


class TestHistoricalFeatures:
    def test_add_historical_features_no_leakage(self):
        from src.features.historical import add_historical_features
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        grid = pd.DataFrame({"cvegeo_inegi": "01001", "date": dates})
        harmonized = pd.DataFrame({
            "cvegeo_inegi": ["01001"],
            "fecha_inicio": [pd.Timestamp("2024-01-03")],
        })
        feat = add_historical_features(grid, harmonized)
        assert "fires_7d" in feat.columns
        # On Jan 3, fire occurs. Jan 3 prediction MUST NOT count Jan 3 fire in fires_7d!
        jan3_fires7d = feat[feat["date"] == "2024-01-03"]["fires_7d"].iloc[0]
        assert jan3_fires7d == 0.0
        # On Jan 4, fires_7d SHOULD count the Jan 3 fire (1.0)
        jan4_fires7d = feat[feat["date"] == "2024-01-04"]["fires_7d"].iloc[0]
        assert jan4_fires7d == 1.0


class TestCalibrateRisk:
    def test_classify_risk_low(self):
        from src.models.calibrate import classify_risk
        thresholds = {"medium_threshold": 0.01, "high_threshold": 0.05}
        assert classify_risk(0.005, thresholds) == "LOW"

    def test_classify_risk_medium(self):
        from src.models.calibrate import classify_risk
        thresholds = {"medium_threshold": 0.01, "high_threshold": 0.05}
        assert classify_risk(0.02, thresholds) == "MEDIUM"

    def test_classify_risk_high(self):
        from src.models.calibrate import classify_risk
        thresholds = {"medium_threshold": 0.01, "high_threshold": 0.05}
        assert classify_risk(0.06, thresholds) == "HIGH"
