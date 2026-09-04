"""
src/data/harmonize.py — Merge both wildfire datasets into a single
unified schema keyed by proper INEGI CVEGEO codes.

Phase 3 entry point:
    python -m src.data.harmonize

Output:
    data/interim/fires_harmonized.parquet
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd
import numpy as np

from src.config import (
    FIRES_HARMONIZED,
    INEGI_MUNICIPALITIES,
    MEX_LAT_MIN,
    MEX_LAT_MAX,
    MEX_LON_MIN,
    MEX_LON_MAX,
    RANDOM_SEED,
)
from src.data.load import load_raw_2015_2024, load_raw_2025

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ── INEGI state-name → numeric code lookup ─────────────────────────────────
# Official codes from INEGI Marco Geoestadístico 2024
STATE_NAME_TO_CODE: dict[str, int] = {
    "aguascalientes": 1,
    "baja california": 2,
    "baja california sur": 3,
    "campeche": 4,
    "coahuila": 5,
    "coahuila de zaragoza": 5,
    "colima": 6,
    "chiapas": 7,
    "chihuahua": 8,
    "ciudad de mexico": 9,
    "durango": 10,
    "guanajuato": 11,
    "guerrero": 12,
    "hidalgo": 13,
    "jalisco": 14,
    "mexico": 15,
    "michoacan": 16,
    "michoacan de ocampo": 16,
    "morelos": 17,
    "nayarit": 18,
    "nuevo leon": 19,
    "oaxaca": 20,
    "puebla": 21,
    "queretaro": 22,
    "queretaro de arteaga": 22,
    "quintana roo": 23,
    "san luis potosi": 24,
    "sinaloa": 25,
    "sonora": 26,
    "tabasco": 27,
    "tamaulipas": 28,
    "tlaxcala": 29,
    "veracruz": 30,
    "veracruz de ignacio de la llave": 30,
    "yucatan": 31,
    "zacatecas": 32,
}

# Prefix-based fallback for state names that survive garbled encoding
# (first 5+ unambiguous ASCII characters of each state name)
_STATE_PREFIX_MAP: dict[str, int] = {
    "aguasc": 1,
    "baja c": 2,   # matched further below with "sur" check
    "campe": 4,
    "coahu": 5,
    "colim": 6,
    "chiap": 7,
    "chihu": 8,
    "ciudad": 9,
    "duran": 10,
    "guanaj": 11,
    "guerre": 12,
    "hidalg": 13,
    "jalis": 14,
    "morelo": 17,
    "nayar": 18,
    "nuevo": 19,
    "oaxac": 20,
    "puebl": 21,
    "quint": 23,   # Quintana Roo
    "sinalo": 25,
    "sonor": 26,
    "tabas": 27,
    "tamau": 28,
    "tlaxc": 29,
    "veracr": 30,
    "zacate": 32,
}

# ── Cause normalization map ───────────────────────────────────────────────────
_CAUSE_MAP = {
    "0": "desconocidas",
    "ninguna / no aplica": "desconocidas",
}

# ── Impact category normalization ─────────────────────────────────────────────
_IMPACT_MAP = {
    "impacto mnimo": "impacto minimo",   # encoding artifact variant
}


def _normalize_text(s: str) -> str:
    """Lowercase, strip, remove accents, and resolve encoding artifacts / aliases."""
    if not isinstance(s, str):
        return s
    s = s.strip().lower()
    # Resolve known Latin-1/CP1252 mojibake sequences
    s = (s
        .replace("a\xa9", "e")
        .replace("a\xa1", "a")
        .replace("a\xb3", "o")
        .replace("a\xad", "i")
        .replace("\ufffd", "")
    )
    # English state aliases
    s = s.replace("mexico city", "ciudad de mexico").replace("state of mexico", "mexico")
    # Remove diacritics
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s


def _dms_to_decimal(degrees: pd.Series, minutes: pd.Series, seconds: pd.Series) -> pd.Series:
    """Convert degrees/minutes/seconds to decimal degrees."""
    return degrees + minutes / 60.0 + seconds / 3600.0


def _reconstruct_inegi_cvegeo(cve_ent: pd.Series, cve_mun: pd.Series) -> pd.Series:
    """Build the standard 5-digit INEGI CVEGEO string: e.g. '01003'."""
    ent = cve_ent.astype(int)
    mun = cve_mun.round().astype(int)   # handles float-stored integers
    return ent.map(lambda e: f"{e:02d}") + mun.map(lambda m: f"{m:03d}")


def _state_name_to_code(name: pd.Series) -> pd.Series:
    """Map normalized state name to INEGI numeric code.

    Uses two passes:
    1. Exact match after normalization (handles ASCII-clean names).
    2. Prefix-based match for names garbled by encoding artifacts
       (e.g., 'ma\x83xico' for 'México').
    Special cases handled explicitly: México(15), Michoacán(16),
    San Luis Potosí(24), Querétaro(22), Yucatán(31), Nuevo León(19).
    """
    def _lookup(raw: str) -> int | None:
        if not isinstance(raw, str):
            return None

        norm = _normalize_text(raw)

        # Pass 1: exact match
        code = STATE_NAME_TO_CODE.get(norm)
        if code is not None:
            return code

        # Pass 2: ASCII-only prefix match
        ascii_only = norm.encode("ascii", errors="ignore").decode("ascii").strip()

        # Special cases that are single common words but collision-prone
        # "mexico" alone → 15 (Estado de México); CDMX starts with "ciudad"
        if ascii_only == "mxico" or ascii_only.startswith("m") and "xico" in norm:
            return 15  # Estado de México
        if "michoa" in ascii_only or ("micho" in ascii_only):
            return 16
        if "san luis" in ascii_only or ("san" in ascii_only and "luis" in ascii_only):
            return 24
        if "quer" in ascii_only and "taro" in ascii_only:
            return 22
        if "yuca" in ascii_only:
            return 31
        if "nuevo" in ascii_only and "le" in ascii_only:
            return 19
        if "ciudad" in ascii_only:
            return 9

        # Pass 3: prefix map
        for prefix, code in _STATE_PREFIX_MAP.items():
            if ascii_only.startswith(prefix):
                # Disambiguate Baja California vs Baja California Sur
                if prefix == "baja c":
                    return 3 if "sur" in ascii_only else 2
                return code

        return None

    return name.map(_lookup)



def _validate_coordinates(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """Return boolean mask: True where coordinates are within Mexico bounds."""
    return (
        lat.between(MEX_LAT_MIN, MEX_LAT_MAX)
        & lon.between(MEX_LON_MIN, MEX_LON_MAX)
    )


# ── 2015-2024 processing ──────────────────────────────────────────────────────

def _process_2015_2024(df: pd.DataFrame, inegi: pd.DataFrame) -> pd.DataFrame:
    """Clean and reformat the 2015-2024 dataset into the unified schema."""
    log.info("Processing 2015-2024 dataset (%d rows)", len(df))

    # Drop purely administrative / fn_ columns
    fn_cols = [c for c in df.columns if c.startswith("fn_")]
    df = df.drop(columns=fn_cols + ["Predio", "Entidad"], errors="ignore")

    # ── Coordinates ──────────────────────────────────────────────────────────
    df["latitud"] = pd.to_numeric(df["latitud"], errors="coerce")
    df["longitud"] = pd.to_numeric(df["longitud"], errors="coerce")

    # Impute invalid coords from INEGI municipality centroid
    invalid_mask = ~_validate_coordinates(df["latitud"], df["longitud"])
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        log.warning("%d rows have invalid coordinates — attempting imputation from INEGI centroid", n_invalid)
        df = _impute_coords_from_inegi(df, inegi, invalid_mask, ent_col="CVE_ENT", mun_col="CVE_MUN")

    # ── Dates ────────────────────────────────────────────────────────────────
    df["fecha_inicio"] = pd.to_datetime(df["Fecha_Inicio"], errors="coerce")
    df["fecha_fin"] = pd.to_datetime(df["Fecha_Termino"], errors="coerce")

    # Fix negative durations (end before start)
    neg_dur = df["fecha_fin"] < df["fecha_inicio"]
    if neg_dur.any():
        log.warning("Swapping %d rows with end-before-start dates", neg_dur.sum())
        df.loc[neg_dur, ["fecha_inicio", "fecha_fin"]] = (
            df.loc[neg_dur, ["fecha_fin", "fecha_inicio"]].values
        )

    # ── INEGI CVEGEO ─────────────────────────────────────────────────────────
    valid_mun = df["CVE_MUN"].notna()
    df["cvegeo_inegi"] = pd.NA
    df.loc[valid_mun, "cvegeo_inegi"] = _reconstruct_inegi_cvegeo(
        df.loc[valid_mun, "CVE_ENT"],
        df.loc[valid_mun, "CVE_MUN"],
    )

    # ── Categorical normalization ─────────────────────────────────────────────
    cat_cols = {
        "Causa": "causa",
        "Causa_especifica": "causa_especifica",
        "Tipo_de_incendio": "tipo_incendio",
        "Tipo_Vegetacion": "tipo_vegetacion",
        "Regimen_de_fuego": "regimen_fuego",
        "Tipo_impacto": "tipo_impacto",
        "Region": "region",
        "Estado": "estado",
        "Municipio": "municipio",
        "Tamano": "tamano",
    }
    for old, new in cat_cols.items():
        if old in df.columns:
            df[new] = df[old].map(_normalize_text)

    # Apply cause correction map
    df["causa"] = df["causa"].replace(_CAUSE_MAP)

    # ── Unified schema ────────────────────────────────────────────────────────
    df = df.rename(columns={
        "anio": "anio",
        "Clave_del_incendio": "clave_incendio",
        "CVE_ENT": "cve_ent",
        "CVE_MUN": "cve_mun",
        "latitud": "latitud",
        "longitud": "longitud",
        "Total_hectareas": "total_ha",
        "Arbolado_Adulto": "arbolado_adulto",
        "Renuevo": "renuevo",
        "Arbustivo": "arbustivo",
        "Herbaceo": "herbaceo",
        "Hojarasca": "hojarasca",
        "Duracion_dias": "duracion_categoria",
    })

    df["dataset"] = "2015_2024"

    # ── Select unified columns ────────────────────────────────────────────────
    keep = [
        "dataset", "anio", "clave_incendio", "cvegeo_inegi",
        "cve_ent", "cve_mun", "estado", "municipio", "region",
        "latitud", "longitud",
        "fecha_inicio", "fecha_fin",
        "causa", "causa_especifica",
        "tipo_incendio", "tipo_vegetacion", "regimen_fuego",
        "tipo_impacto", "tamano", "duracion_categoria",
        "total_ha", "arbolado_adulto", "renuevo", "arbustivo",
        "herbaceo", "hojarasca",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# ── 2025 processing ───────────────────────────────────────────────────────────

def _process_2025(df: pd.DataFrame, inegi: pd.DataFrame) -> pd.DataFrame:
    """Clean and reformat the 2025 dataset into the unified schema."""
    log.info("Processing 2025 dataset (%d rows)", len(df))

    # ── Coordinates — convert DMS to decimal degrees ─────────────────────────
    df["latitud"] = _dms_to_decimal(
        df["latitud_grados"].astype(float),
        df["latitud_minutos"].astype(float),
        df["latitud_segundos"].astype(float),
    )
    # Mexico is in the Western Hemisphere; longitude should be negative
    df["longitud"] = -_dms_to_decimal(
        df["longitud_grados"].astype(float),
        df["longitud_minutos"].astype(float),
        df["longitud_segundos"].astype(float),
    )

    # Validate
    invalid_mask = ~_validate_coordinates(df["latitud"], df["longitud"])
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        log.warning("%d rows have invalid coordinates in 2025 — attempting imputation", n_invalid)
        df = _impute_coords_from_inegi(df, inegi, invalid_mask, ent_col="_cve_ent", mun_col="cve_municipio")

    # ── State code ────────────────────────────────────────────────────────────
    df["cve_ent"] = _state_name_to_code(df["entidad"])
    unmapped = df["cve_ent"].isna().sum()
    if unmapped > 0:
        log.warning("%d rows in 2025 have unrecognized state names", unmapped)

    # ── INEGI CVEGEO ─────────────────────────────────────────────────────────
    valid = df["cve_ent"].notna() & df["cve_municipio"].notna()
    df["cvegeo_inegi"] = pd.NA
    df.loc[valid, "cvegeo_inegi"] = _reconstruct_inegi_cvegeo(
        df.loc[valid, "cve_ent"],
        df.loc[valid, "cve_municipio"],
    )

    # ── Dates ────────────────────────────────────────────────────────────────
    df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce")
    df["fecha_fin"] = pd.to_datetime(df["fecha_liquidacion"], errors="coerce")

    # ── Categoricals ──────────────────────────────────────────────────────────
    cat_renames = {
        "entidad": "estado",
        "municipio": "municipio",
        "posible_causa": "causa",
        "posible_causa_especifica": "causa_especifica",
        "tipo_incendio": "tipo_incendio",
        "tipo_vegetacion": "tipo_vegetacion",
        "regimen_fuego": "regimen_fuego",
        "clasf_primer_orden": "tipo_impacto",
        "clasificacion_sup_afectada": "tamano",
        "categoria_duracion_dias": "duracion_categoria",
        "zona_bioclimatica": "zona_bioclimatica",
        "crmf": "region",
        "cve_municipio": "cve_mun",
    }
    df = df.rename(columns=cat_renames)

    norm_cols = ["causa", "causa_especifica", "tipo_incendio", "tipo_vegetacion",
                 "regimen_fuego", "tipo_impacto", "estado", "municipio",
                 "tamano", "region", "zona_bioclimatica"]
    for c in norm_cols:
        if c in df.columns:
            df[c] = df[c].map(_normalize_text)

    df["causa"] = df["causa"].replace(_CAUSE_MAP)
    df["dataset"] = "2025"
    df["anio"] = df["fecha_inicio"].dt.year.fillna(2025).astype("int16")

    keep = [
        "dataset", "anio", "clave_incendio", "cvegeo_inegi",
        "cve_ent", "cve_mun", "estado", "municipio", "region",
        "latitud", "longitud",
        "fecha_inicio", "fecha_fin",
        "causa", "causa_especifica",
        "tipo_incendio", "tipo_vegetacion", "regimen_fuego",
        "tipo_impacto", "tamano", "duracion_categoria",
        "zona_bioclimatica",
        "total_ha", "arbolado_adulto", "renuevo", "arbustivo",
        "herbaceo", "hojarasca",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


# ── Coordinate imputation helper ──────────────────────────────────────────────

def _impute_coords_from_inegi(
    df: pd.DataFrame,
    inegi: pd.DataFrame,
    invalid_mask: pd.Series,
    ent_col: str,
    mun_col: str,
) -> pd.DataFrame:
    """Replace invalid coordinates with INEGI municipality centroids."""
    if inegi is None or inegi.empty:
        log.warning("INEGI catalog not available — dropping %d rows with invalid coords", invalid_mask.sum())
        return df[~invalid_mask].copy()

    inegi_lookup = inegi.set_index("cvegeo_inegi")[["lat_centroid", "lon_centroid"]]

    for idx in df[invalid_mask].index:
        try:
            ent = int(df.at[idx, ent_col] if ent_col in df.columns else 0)
            mun = int(df.at[idx, mun_col]) if df.at[idx, mun_col] == df.at[idx, mun_col] else 0
            key = f"{ent:02d}{mun:03d}"
            if key in inegi_lookup.index:
                df.at[idx, "latitud"] = inegi_lookup.at[key, "lat_centroid"]
                df.at[idx, "longitud"] = inegi_lookup.at[key, "lon_centroid"]
        except Exception:
            pass

    # Drop rows still invalid after imputation
    still_invalid = ~_validate_coordinates(df["latitud"], df["longitud"])
    n_drop = still_invalid.sum()
    if n_drop > 0:
        log.warning("Dropping %d rows whose coordinates could not be imputed", n_drop)
        df = df[~still_invalid].copy()

    return df


# ── Backfill zona_bioclimatica from 2025 ─────────────────────────────────────

def _backfill_bioclimatic_zone(harmonized: pd.DataFrame) -> pd.DataFrame:
    """Map zona_bioclimatica from 2025 records to 2015-2024 records.

    Since bioclimatic zones are a static geographic property, the most
    frequent zone observed per municipality in 2025 is used as the lookup.
    """
    if "zona_bioclimatica" not in harmonized.columns:
        return harmonized

    # Build lookup from 2025 data
    zona_2025 = (
        harmonized[harmonized["dataset"] == "2025"]
        .groupby("cvegeo_inegi")["zona_bioclimatica"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NA)
    )

    # Fill missing values in 2015-2024 rows
    mask_missing = harmonized["zona_bioclimatica"].isna() & (harmonized["dataset"] == "2015_2024")
    harmonized.loc[mask_missing, "zona_bioclimatica"] = (
        harmonized.loc[mask_missing, "cvegeo_inegi"].map(zona_2025)
    )

    covered = harmonized["zona_bioclimatica"].notna().sum()
    total = len(harmonized)
    log.info("zona_bioclimatica coverage after backfill: %d/%d (%.1f%%)", covered, total, 100 * covered / total)
    return harmonized


# ── 2015 completeness check ───────────────────────────────────────────────────

def _check_2015_completeness(df: pd.DataFrame) -> bool:
    """Return True if 2015 data appears complete (full-year coverage).

    Criteria: at least one fire record for every month Jan–Dec 2015.
    If any month is missing, the year is considered incomplete → exclude.
    """
    fires_2015 = df[df["anio"] == 2015].copy()
    if fires_2015.empty:
        return False
    months_present = fires_2015["fecha_inicio"].dt.month.dropna().unique()
    is_complete = len(months_present) == 12
    if not is_complete:
        log.warning(
            "2015 data appears INCOMPLETE — only %d/12 months present (%s). "
            "Excluding 2015 from training as per team decision.",
            len(months_present),
            sorted(months_present.tolist()),
        )
    else:
        log.info("2015 data appears complete (all 12 months present).")
    return is_complete


# ── Main ──────────────────────────────────────────────────────────────────────

def harmonize(save: bool = True) -> pd.DataFrame:
    """Full harmonization pipeline.

    Loads both raw datasets, cleans, merges, and optionally saves to parquet.

    Returns
    -------
    pd.DataFrame
        Unified fire-event DataFrame.
    """
    # Load INEGI catalog if available (for coordinate imputation)
    inegi = None
    if INEGI_MUNICIPALITIES.exists():
        inegi = pd.read_csv(INEGI_MUNICIPALITIES, dtype={"cvegeo_inegi": str})
        log.info("INEGI catalog loaded: %d municipalities", len(inegi))
    else:
        log.warning(
            "INEGI catalog not found at %s — coordinate imputation will be skipped. "
            "Run src/data/weather.py (which downloads the catalog) first.",
            INEGI_MUNICIPALITIES,
        )

    # Load raw data
    raw_2024 = load_raw_2015_2024()
    raw_2025 = load_raw_2025()

    # Process each dataset
    df_2024 = _process_2015_2024(raw_2024, inegi)
    df_2025 = _process_2025(raw_2025, inegi)

    # Exclude 2025 records that actually started in 2024
    n_pre2025 = (df_2025["fecha_inicio"].dt.year < 2025).sum()
    if n_pre2025 > 0:
        log.info("Moving %d 2025-file records with 2024 start dates into 2024 pool", n_pre2025)
        df_pre2025 = df_2025[df_2025["fecha_inicio"].dt.year < 2025].copy()
        df_2025 = df_2025[df_2025["fecha_inicio"].dt.year >= 2025].copy()
        df_2024 = pd.concat([df_2024, df_pre2025], ignore_index=True)

    # Check 2015 completeness and exclude if needed
    is_2015_complete = _check_2015_completeness(df_2024)
    if not is_2015_complete:
        df_2024 = df_2024[df_2024["anio"] != 2015].copy()
        log.info("2015 excluded. Remaining 2015-2024 rows: %d", len(df_2024))

    # Merge
    harmonized = pd.concat([df_2024, df_2025], ignore_index=True)

    # Backfill bioclimatic zone from 2025 data
    harmonized = _backfill_bioclimatic_zone(harmonized)

    # Sort by date
    harmonized = harmonized.sort_values("fecha_inicio").reset_index(drop=True)

    log.info(
        "Harmonized dataset: %d total fire events | %d unique CVEGEO municipalities",
        len(harmonized),
        harmonized["cvegeo_inegi"].nunique(),
    )

    if save:
        FIRES_HARMONIZED.parent.mkdir(parents=True, exist_ok=True)
        harmonized.to_parquet(FIRES_HARMONIZED, index=False)
        log.info("Saved to %s", FIRES_HARMONIZED)

    return harmonized


if __name__ == "__main__":
    harmonize()
