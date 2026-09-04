"""
src/models/calibrate.py — Post-hoc probability calibration.

Applies Platt scaling (sigmoid) or isotonic regression calibration
if ECE > 0.05 on the validation set.

Also computes and saves quantile-based risk thresholds (team decision:
90th / 99th percentile of validation predictions → MEDIUM / HIGH).

Usage:
    python -m src.models.calibrate
"""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.config import (
    CALIBRATOR,
    FEATURES_VAL,
    MODEL_LGBM,
    MODEL_LR,
    MODEL_RF,
    PREPROCESSOR,
    THRESHOLDS,
    MEDIUM_RISK_PERCENTILE,
    HIGH_RISK_PERCENTILE,
    RANDOM_SEED,
)
from src.features.static import TARGET_COL
from src.models.evaluate import evaluate_model, _compute_ece
from src.models.train import get_feature_cols, prepare_xy

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

ECE_THRESHOLD = 0.05   # Apply calibration only if ECE exceeds this


def calibrate_best_model(
    model,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_name: str = "best_model",
) -> object:
    """Apply post-hoc calibration if needed.

    Uses isotonic regression (robust, non-parametric).
    Falls back to Platt scaling (sigmoid) for small validation sets.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Trained model.
    X_val : pd.DataFrame
        Validation features.
    y_val : pd.Series
        Validation labels.
    model_name : str
        Name for logging.

    Returns
    -------
    Calibrated or original model.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    ece_before = _compute_ece(y_val.values, y_prob)
    log.info("[%s] ECE before calibration: %.4f", model_name, ece_before)

    if ece_before <= ECE_THRESHOLD:
        log.info("ECE ≤ %.2f — calibration not needed.", ECE_THRESHOLD)
        return model

    log.info("ECE > %.2f — applying isotonic regression calibration.", ECE_THRESHOLD)
    try:
        from sklearn.frozen import FrozenEstimator
        calibrated = CalibratedClassifierCV(FrozenEstimator(model), method="isotonic")
    except ImportError:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_val, y_val)

    y_prob_cal = calibrated.predict_proba(X_val)[:, 1]
    ece_after = _compute_ece(y_val.values, y_prob_cal)
    log.info("[%s] ECE after calibration: %.4f", model_name, ece_after)

    return calibrated


def compute_risk_thresholds(
    model,
    X_val: pd.DataFrame,
) -> dict:
    """Compute quantile-based LOW/MEDIUM/HIGH risk thresholds.

    Team decision: 90th percentile → MEDIUM, 99th → HIGH.
    Thresholds are computed from the distribution of all validation
    predicted probabilities.

    Returns
    -------
    dict
        {'medium_threshold': float, 'high_threshold': float}
    """
    y_prob = model.predict_proba(X_val)[:, 1]

    medium_thresh = float(np.percentile(y_prob, MEDIUM_RISK_PERCENTILE))
    high_thresh = float(np.percentile(y_prob, HIGH_RISK_PERCENTILE))

    thresholds = {
        "medium_threshold": medium_thresh,
        "high_threshold": high_thresh,
        "medium_percentile": MEDIUM_RISK_PERCENTILE,
        "high_percentile": HIGH_RISK_PERCENTILE,
        "n_validation_rows": len(y_prob),
        "prob_min": float(y_prob.min()),
        "prob_max": float(y_prob.max()),
        "prob_mean": float(y_prob.mean()),
    }

    log.info(
        "Risk thresholds — MEDIUM: p > %.6f | HIGH: p > %.6f",
        medium_thresh, high_thresh,
    )
    return thresholds


def classify_risk(probability: float, thresholds: dict) -> str:
    """Convert a probability to a risk level string.

    Parameters
    ----------
    probability : float
        Calibrated predicted probability (0–1).
    thresholds : dict
        Output of compute_risk_thresholds().

    Returns
    -------
    str
        'LOW', 'MEDIUM', or 'HIGH'.
    """
    if probability >= thresholds["high_threshold"]:
        return "HIGH"
    elif probability >= thresholds["medium_threshold"]:
        return "MEDIUM"
    return "LOW"


def main() -> None:
    log.info("Loading validation split...")
    val = pd.read_parquet(FEATURES_VAL)
    meta = joblib.load(PREPROCESSOR)
    feature_cols = meta["feature_cols"]

    X_val, y_val = prepare_xy(val, feature_cols)

    # Determine best model by PR-AUC (simple comparison)
    model_paths = {
        "Logistic Regression": MODEL_LR,
        "Random Forest": MODEL_RF,
        "LightGBM": MODEL_LGBM,
    }

    best_name = None
    best_prauc = -1.0
    best_model = None

    for name, path in model_paths.items():
        if not path.exists():
            log.warning("Model not found: %s — skipping", path)
            continue
        model = joblib.load(path)
        y_prob = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import average_precision_score
        prauc = average_precision_score(y_val, y_prob)
        log.info("[%s] Validation PR-AUC: %.4f", name, prauc)
        if prauc > best_prauc:
            best_prauc = prauc
            best_name = name
            best_model = model

    if best_model is None:
        raise RuntimeError("No trained models found. Run src.models.train first.")

    log.info("Best model: %s (PR-AUC=%.4f)", best_name, best_prauc)

    # Calibrate best model
    calibrated = calibrate_best_model(best_model, X_val, y_val, best_name)

    # Save calibrated model
    CALIBRATOR.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": calibrated, "name": best_name, "feature_cols": feature_cols}, CALIBRATOR)
    log.info("Calibrated model saved to %s", CALIBRATOR)

    # Compute and save risk thresholds
    thresholds = compute_risk_thresholds(calibrated, X_val)
    THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLDS, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("Risk thresholds saved to %s", THRESHOLDS)


if __name__ == "__main__":
    main()
