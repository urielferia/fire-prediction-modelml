"""
src/models/explain.py — SHAP-based model explainability.

Computes SHAP values for the best calibrated model and generates:
  - Global summary (beeswarm) plot
  - Feature importance bar chart
  - Dependence plots for top features
  - Example local (waterfall) explanation

Usage:
    python -m src.models.explain
"""

from __future__ import annotations

import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.config import CALIBRATOR, FEATURES_VAL, FIGURES_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Max rows to use for SHAP computation (subsample for speed)
MAX_SHAP_ROWS = 5000


def compute_shap_values(
    model,
    X: pd.DataFrame,
    model_name: str = "best_model",
) -> shap.Explanation:
    """Compute SHAP values using the appropriate explainer for the model type.

    For LightGBM: uses TreeExplainer (fast, exact).
    For Logistic Regression / others: uses LinearExplainer or KernelExplainer.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Trained (and optionally calibrated) model.
    X : pd.DataFrame
        Feature matrix to explain.
    model_name : str
        Name for logging and filename generation.

    Returns
    -------
    shap.Explanation
        SHAP values object.
    """
    log.info("Computing SHAP values for %s on %d rows...", model_name, len(X))

    # Subsample for speed
    if len(X) > MAX_SHAP_ROWS:
        X = X.sample(n=MAX_SHAP_ROWS, random_state=42)
        log.info("Subsampled to %d rows for SHAP computation.", MAX_SHAP_ROWS)

    # Unwrap CalibratedClassifierCV to get the base estimator
    base_model = _unwrap_calibrated(model)

    try:
        import lightgbm as lgb
        if isinstance(base_model, lgb.LGBMClassifier):
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer(X)
            # For binary classification, TreeExplainer returns shape (n, p, 2)
            # — take class-1 values
            if isinstance(shap_values, shap.Explanation) and len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            return shap_values
    except ImportError:
        pass

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    if isinstance(base_model, RandomForestClassifier):
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer(X)
        if isinstance(shap_values, shap.Explanation) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        return shap_values

    if isinstance(base_model, LogisticRegression):
        # LinearExplainer works well for logistic regression
        explainer = shap.LinearExplainer(base_model, X)
        shap_values = explainer(X)
        return shap_values

    # Fallback: KernelExplainer (slow but model-agnostic)
    log.warning("Using KernelExplainer — this may be slow. Consider using a tree-based model.")
    background = shap.sample(X, min(100, len(X)))
    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(pd.DataFrame(x, columns=X.columns))[:, 1],
        background,
    )
    shap_vals = explainer.shap_values(X)
    return shap_vals


def _unwrap_calibrated(model) -> object:
    """Extract the underlying estimator from a CalibratedClassifierCV."""
    from sklearn.calibration import CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        est = model.calibrated_classifiers_[0].estimator
        try:
            from sklearn.frozen import FrozenEstimator
            if isinstance(est, FrozenEstimator):
                est = est.estimator
        except ImportError:
            pass
        return est
    return model


def plot_shap_summary(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    model_name: str = "best_model",
    max_display: int = 20,
) -> None:
    """Generate and save SHAP beeswarm summary plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values if not isinstance(shap_values, shap.Explanation) else shap_values.values,
        X,
        max_display=max_display,
        show=False,
    )
    path = FIGURES_DIR / f"shap_summary_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("SHAP summary plot saved to %s", path)


def plot_shap_importance(
    shap_values: shap.Explanation,
    X: pd.DataFrame,
    model_name: str = "best_model",
    top_n: int = 20,
) -> pd.DataFrame:
    """Generate feature importance table and bar chart from SHAP values."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    vals = shap_values.values if isinstance(shap_values, shap.Explanation) else shap_values
    mean_abs_shap = np.abs(vals).mean(axis=0)

    importance = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    # Save table
    importance.to_csv(
        RESULTS_DIR / f"shap_importance_{model_name.lower().replace(' ', '_')}.csv",
        index=False,
    )

    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance["feature"][::-1], importance["mean_abs_shap"][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Top {top_n} Features by SHAP Importance — {model_name}")
    ax.grid(True, axis="x", alpha=0.3)
    path = FIGURES_DIR / f"shap_bar_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("SHAP importance bar chart saved to %s", path)

    log.info("\nTop 10 most important features:\n%s",
             importance.head(10).to_string(index=False))

    return importance


def plot_local_explanation(
    model,
    explainer,
    X_row: pd.DataFrame,
    model_name: str = "best_model",
    label: str = "example",
) -> None:
    """Save waterfall plot for a single prediction (local explanation)."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    shap_val = explainer(X_row)
    if isinstance(shap_val, shap.Explanation) and len(shap_val.shape) == 3:
        shap_val = shap_val[:, :, 1]
    shap.waterfall_plot(shap_val[0], show=False)
    path = FIGURES_DIR / f"shap_waterfall_{label}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Local SHAP explanation saved to %s", path)


def main() -> None:
    log.info("Loading calibrated model and validation data...")

    if not CALIBRATOR.exists():
        raise FileNotFoundError(
            f"Calibrated model not found at {CALIBRATOR}. Run src.models.calibrate first."
        )

    bundle = joblib.load(CALIBRATOR)
    model = bundle["model"]
    model_name = bundle["name"]
    feature_cols = bundle["feature_cols"]

    val = pd.read_parquet(FEATURES_VAL)
    X_val = val[feature_cols]

    # Compute SHAP values
    shap_values = compute_shap_values(model, X_val, model_name)

    # Global summary plot
    X_sample = X_val.sample(n=min(MAX_SHAP_ROWS, len(X_val)), random_state=42)
    plot_shap_summary(shap_values, X_sample, model_name)

    # Feature importance
    plot_shap_importance(shap_values, X_sample, model_name)

    log.info("SHAP analysis complete.")


if __name__ == "__main__":
    main()
