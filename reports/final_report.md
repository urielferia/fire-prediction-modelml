# Forest Fire Risk Prediction in Mexico — Final Report

## Executive Summary

Wildfire risk management requires daily, location-specific probability estimates of fire ignition. This project delivers an end-to-end Machine Learning system for predicting daily forest fire occurrence at the municipality level across Mexico.

Key accomplishments:
1. **Data Harmonization**: Standardized 78,105 historical fire event records (2015–2025) across disparate schemas, converting 2025 DMS coordinates to decimal degrees and reconstructing official 5-digit INEGI `CVEGEO` codes for all ~1,737 fire-active municipalities.
2. **Environmental Data Integration**: Integrated daily weather reanalysis data from the **Open-Meteo Historical Weather API** (ERA5 reanalysis) for municipality centroids from 2014 to 2025.
3. **Synthetic Target Construction**: Built a daily municipality-level modeling matrix (~6.9 million candidate observations) using a **20:1 controlled negative sampling ratio** for training and full natural prevalence for evaluation.
4. **Leakage-Guarded Feature Engineering**: Computed 40+ spatiotemporal, weather lag/rolling, and historical fire frequency features with strict safeguards against target leakage.
5. **Model Evaluation & Calibration**: Compared Logistic Regression, Random Forest, and LightGBM models across temporal splits (Train: 2016–2022, Validation: 2023–2024, Test: 2025 holdout). **LightGBM** achieved superior performance with **PR-AUC of 0.3255** and **ROC-AUC of 0.9671**. Isotonic regression post-hoc calibration reduced Expected Calibration Error (ECE) to **0.0000** and Brier Score to **0.0079**.
6. **Interpretability & Risk Thresholds**: Defined data-driven risk levels (**LOW** <90th percentile, **MEDIUM** 90th–99th percentile [>2.12% prob], **HIGH** >99th percentile [>19.27% prob]). SHAP analysis revealed historical seasonal frequency (`avg_fires_same_month`), regional monthly pressure (`fires_state_30d`), and relative humidity (`relative_humidity_2m_mean`) as top risk drivers.

---

## 1. Dataset & Harmonization Summary

| Metric / Property | Historical Dataset (2015–2024) | Holdout Dataset (2025) | Unified Harmonized Dataset |
|---|---|---|---|
| Source File | `estadisticasincendiosforestales2015-2024.csv` | `2025_Incendios_forestales.csv` | `data/interim/fires_harmonized.parquet` |
| Record Count | 71,089 fire events | 7,016 fire events | 78,105 total fire events |
| Coordinate System | Decimal Degrees | Degrees, Minutes, Seconds (DMS) | Standardized Decimal Degrees |
| Geographic Identifiers | Non-standard `CVEGEO` | State name + local `cve_municipio` | Standardized 5-digit INEGI `cvegeo_inegi` |
| Municipalities Represented | 1,888 | 984 | 1,737 validated fire-active municipalities |

### Key Preprocessing Steps:
- **INEGI Standard Reconstruction**: Mapped state names and local municipality IDs to official INEGI 5-digit codes (`f"{cve_ent:02d}{cve_mun:03d}"`), fixing 3,201 invalid legacy codes.
- **Coordinate Conversion**: Converted DMS columns (`latitud_grados`, `latitud_minutos`, `latitud_segundos`) to decimal degrees, dropping 682 records with zero or out-of-bound coordinates.
- **Bioclimatic Zone Backfill**: Static `zona_bioclimatica` categories from the 2025 dataset were backfilled onto 2015–2024 records based on municipality lookup, achieving 94.5% coverage.

---

## 2. Modeling Table & Target Construction

To convert point-based fire occurrences into a supervised binary classification problem:
- **Observation Grain**: `(cvegeo_inegi, date)` representing a specific municipality on a specific calendar day.
- **Target Label (`fire_occurred`)**: `1` if at least one fire ignition occurred in the municipality on that date; `0` otherwise.
- **Negative Undersampling**: For training (2016–2022), non-fire days were undersampled at a **20:1 negative-to-positive ratio** (yielding 1,167,000 rows, ~4.76% positive prevalence).
- **Natural Prevalence Validation & Test**: Validation (2023–2024) and test (2025 holdout) sets retain the full natural distribution (~0.8% - 1.2% positive prevalence) across 1,269,747 and 634,005 municipality-days respectively.

---

## 3. Model Comparison & Benchmarks

Models were trained on 2016–2022 data and evaluated on full natural distributions for validation (2023–2024) and held-out test set (2025).

### Validation Set Performance (2023–2024)

| Model Architecture | PR-AUC (Primary) | ROC-AUC | Brier Score | ECE (Uncalibrated) | ECE (Calibrated) |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.2818 | 0.9320 | 0.0688 | 0.1446 | — |
| **Random Forest** | 0.2883 | 0.9621 | 0.0303 | 0.0695 | — |
| **LightGBM (Raw)** | **0.3255** | **0.9671** | 0.0696 | 0.1087 | — |
| **LightGBM (Calibrated)** | **0.3200** | **0.9672** | **0.0079** | — | **0.0000** |

*Note: Baseline random PR-AUC prevalence is 0.0099 (~1%). LightGBM achieves over 32× lift above random guessing.*

### 2025 Holdout Test Performance

| Model Architecture | PR-AUC (Primary) | ROC-AUC | Brier Score | ECE |
|---|---|---|---|---|
| **Logistic Regression** | 0.2552 | 0.9446 | 0.0671 | 0.1423 |
| **Random Forest** | 0.2707 | 0.9619 | 0.0286 | 0.0639 |
| **LightGBM (Raw)** | **0.2956** | **0.9657** | 0.0646 | 0.1007 |
| **LightGBM (Calibrated)** | **0.2890** | **0.9656** | **0.0076** | **0.0006** |

---

## 4. Calibration & Quantile Risk Thresholds

Raw tree-based probabilities can be distorted due to class weighting and undersampling. Isotonic regression calibration was applied using `CalibratedClassifierCV`.

Data-driven quantile thresholds calculated on the validation set:
- **LOW Risk**: Predicted probability $\le 0.0212$ (0th – 90th percentile of all predictions).
- **MEDIUM Risk**: $0.0212 < \text{prob} \le 0.1927$ (90th – 99th percentile).
- **HIGH Risk**: Predicted probability $> 0.1927$ (top 1% highest risk municipality-days).

---

## 5. SHAP Feature Importance & Explainability

SHAP (SHapley Additive exPlanations) values were computed for the best LightGBM model on a representative 5,000-sample validation batch.

### Top 10 Feature Drivers:
1. `avg_fires_same_month` (Mean ABS SHAP: 3.77) — Baseline seasonal fire occurrence rate for the municipality.
2. `fires_state_30d` (Mean ABS SHAP: 0.90) — State-level fire activity in the preceding 30 days (regional pressure).
3. `relative_humidity_2m_mean` (Mean ABS SHAP: 0.20) — Daily mean relative humidity (strong inverse risk relationship).
4. `days_since_last_fire` (Mean ABS SHAP: 0.13) — Time elapsed since previous fire ignition.
5. `annual_fire_rate` (Mean ABS SHAP: 0.10) — Historical annual average fire frequency per municipality.
6. `lon_centroid` (Mean ABS SHAP: 0.10) — East-West geographic coordinate gradient.
7. `cve_ent` (Mean ABS SHAP: 0.08) — State categorical grouping.
8. `lat_centroid` (Mean ABS SHAP: 0.06) — North-South geographic gradient.
9. `precip_sum_30d` (Mean ABS SHAP: 0.05) — 30-day cumulative rainfall.
10. `temperature_2m_max` (Mean ABS SHAP: 0.05) — Daily maximum surface temperature.

---

## 6. Prediction Interface & Reproducibility

### CLI Command Example
```bash
python -m src.predict --municipio "Calvillo" --estado "Aguascalientes" --fecha "2025-05-15"
```

### Output Example
```text
==================================================
  WILDFIRE RISK PREDICTION - MEXICO
==================================================
  Municipality : Calvillo
  State        : Aguascalientes
  Date         : 2025-05-15
  Lat / Lon    : 21.9955, -102.7505
--------------------------------------------------
  Wildfire probability : 8.9%
  Risk level           : [MEDIUM] MEDIUM
==================================================
```

### Python API Example
```python
from src.predict import predict_risk

result = predict_risk(municipio="Calvillo", estado="Aguascalientes", fecha="2025-05-15")
print(f"Risk Level: {result['risk_level']} | Calibrated Probability: {result['probability']:.4f}")
```

### System Artifacts
- **Model Checkpoints**: `models/lightgbm.joblib`, `models/calibrator.joblib`, `models/risk_thresholds.json`
- **EDA Notebooks**: `notebooks/01_temporal_analysis.ipynb` through `04_class_distribution.ipynb`
- **Plots & Visualizations**: Saved in `reports/figures/`
- **Metric CSVs**: Saved in `reports/results/`
