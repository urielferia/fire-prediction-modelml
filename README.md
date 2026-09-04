# tayun-fire — Forest Fire Risk Prediction in Mexico

Academic ML project for estimating daily wildfire ignition probability at the municipality level across Mexico.

## Project Structure

```
tayun-fire/
├── data/
│   ├── raw/          # Original untouched CSVs (2015-2024 & 2025)
│   ├── external/     # INEGI catalog, weather cache
│   ├── interim/      # Harmonized / intermediate outputs
│   └── processed/    # Final train/val/test parquet files
├── src/
│   ├── config.py     # All paths, seeds, constants
│   ├── data/         # Harmonization, cleaning, weather, dataset building
│   ├── features/     # Feature engineering modules (temporal, weather, historical, static)
│   └── models/       # Training, evaluation, calibration, explainability
├── notebooks/        # EDA Jupyter notebooks (01_temporal through 04_class_distribution)
├── static/           # Interactive web map assets (HTML, CSS, JS)
├── docs/             # Application and pipeline documentation
├── models/           # Serialized trained models & calibrator
├── reports/          # Visualizations, comparison CSVs, final_report.md
└── tests/            # Pytest test suite
```

## Interactive Map Web Application

An interactive web map built with Flask and Leaflet.js (CartoDB Positron light basemap). Users can select any Mexican state and municipality, query any calendar date up to **today + 15 days**, and view animated predictions with color-coded risk levels.

```bash
# Launch the interactive web server
python app.py --debug
```
Open **http://127.0.0.1:5000** in your web browser.

> 📖 **Full Application Documentation**: See [docs/interactive_map.md](docs/interactive_map.md) for architecture, API specifications, and design system tokens.

## Quick Start — CLI Prediction

To predict the wildfire probability for a specific municipality and date:

```bash
python -m src.predict --municipio "Calvillo" --estado "Aguascalientes" --fecha "2025-05-15"
```

Example CLI Output:
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

## Python API Usage

```python
from src.predict import predict_risk

result = predict_risk(municipio="Calvillo", estado="Aguascalientes", fecha="2025-05-15")
print(result)
# {'municipio': 'Calvillo', 'estado': 'Aguascalientes', 'date': '2025-05-15', 'probability': 8.9, 'risk_level': 'MEDIUM', ...}
```

## Running Tests

```bash
python -m pytest
```

---

## Full Pipeline Reproduction (From Scratch)

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Pipeline Execution Steps

```bash
# Step 1: Download & cache ERA5 daily weather from Open-Meteo
python -m src.data.weather

# Step 2: Harmonize historical (2015-2024) and 2025 datasets into unified schema
python -m src.data.harmonize

# Step 3: Build daily municipality modeling table (20:1 negative sampling for train)
python -m src.data.build_dataset

# Step 4: Build feature matrices (temporal, weather lags, historical fire rates)
python -m src.features.build_features

# Step 5: Train models (Logistic Regression, Random Forest, LightGBM)
python -m src.models.train --model all

# Step 6: Perform probability calibration & risk threshold calculation
python -m src.models.calibrate

# Step 7: Compute SHAP explainability plots
python -m src.models.explain
```

---

## Exploratory Data Analysis (EDA) Notebooks

Launch Jupyter to inspect the analysis notebooks in [`notebooks/`](file:///c:/Users/purpl/tayun-fire/notebooks/):

```bash
jupyter notebook notebooks/
```

- `01_temporal_analysis.ipynb` — Annual trends, monthly seasonality, day-of-week patterns.
- `02_geographic_analysis.ipynb` — State and municipality fire volume rankings, coordinate clustering.
- `03_weather_correlations.ipynb` — Environmental feature distributions (temp, precip, dry streaks).
- `04_class_distribution.ipynb` — Class imbalance analysis across splits.

---

## Temporal Split & Model Benchmarks

| Split | Period | Primary Metric (PR-AUC) | ROC-AUC | Brier Score | ECE |
|---|---|---|---|---|---|
| Train | 2016-01-01 – 2022-12-31 | — | — | — | — |
| Validation | 2023-01-01 – 2024-12-31 | **0.3200** | **0.9672** | **0.0079** | **0.0000** |
| Test (2025 Holdout) | 2025-01-01 – 2025-12-31 | **0.2890** | **0.9656** | **0.0076** | **0.0006** |

*Model: LightGBM (Isotonic Calibrated)*

---

## Risk Levels

Quantile thresholds derived on validation predictions:
- **LOW**: Probability $\le 2.12\%$ (0–90th percentile)
- **MEDIUM**: $2.12\% < \text{prob} \le 19.27\%$ (90th–99th percentile)
- **HIGH**: Probability $> 19.27\%$ (99th percentile +)

---

## Final Report

For a complete breakdown of methodology, data engineering, calibration curves, SHAP interpretability, and evaluation benchmarks, view [`reports/final_report.md`](file:///c:/Users/purpl/tayun-fire/reports/final_report.md).
