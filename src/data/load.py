"""
src/data/load.py — Load raw CSV datasets into DataFrames.

Usage:
    from src.data.load import load_raw_2015_2024, load_raw_2025
"""

import pandas as pd
from src.config import RAW_2015_2024, RAW_2025

# Encoding used by both original files
_ENCODING = "latin-1"

# ── 2015-2024 column dtypes ───────────────────────────────────────────────────
_DTYPE_2024 = {
    "anio": "int16",
    "CVE_ENT": "int8",
    "CVEGEO": "int32",
}


def load_raw_2015_2024() -> pd.DataFrame:
    """Load the 2015–2024 wildfire CSV as-is (no cleaning applied).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with 41 columns and up to 71,089 rows.

    Notes
    -----
    Encoding is latin-1. Dates remain as strings; call harmonize.py to
    parse and clean them.
    """
    df = pd.read_csv(
        RAW_2015_2024,
        encoding=_ENCODING,
        dtype={
            "anio": "int16",
            "CVE_ENT": "int8",
            "CVEGEO": "int32",
        },
        low_memory=False,
    )
    return df


def load_raw_2025() -> pd.DataFrame:
    """Load the 2025 wildfire CSV as-is (no cleaning applied).

    Returns
    -------
    pd.DataFrame
        Raw DataFrame with 76 columns and up to 7,016 rows.

    Notes
    -----
    Coordinates are stored in DMS (degrees/minutes/seconds) format in
    separate columns. Use harmonize.py to convert to decimal degrees.
    """
    df = pd.read_csv(
        RAW_2025,
        encoding=_ENCODING,
        low_memory=False,
    )
    return df
