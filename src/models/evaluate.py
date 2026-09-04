"""
src/models/evaluate.py — Model evaluation: metrics, calibration curves,
confusion matrix, and comparison table.

Usage:
    from src.models.evaluate import evaluate_model, compare_models
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)

from src.config import FIGURES_DIR, RESULTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "model",
    threshold: float = 0.5,
    save_plots: bool = True,
) -> dict:
    """Compute all evaluation metrics for a trained model.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Trained model with predict_proba method.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        True binary labels.
    model_name : str
        Name used for plot titles and filenames.
    threshold : float
        Classification threshold for precision/recall/F1.
    save_plots : bool
        If True, save calibration and PR curve plots.

    Returns
    -------
    dict
        Dictionary of metric name → value.
    """
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {}

    # Core probability metrics
    metrics["roc_auc"] = roc_auc_score(y, y_prob)
    metrics["pr_auc"] = average_precision_score(y, y_prob)
    metrics["brier_score"] = brier_score_loss(y, y_prob)

    # At chosen threshold
    metrics["threshold"] = threshold
    metrics["f1"] = f1_score(y, y_pred, zero_division=0)

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics["tn"] = int(tn)
    metrics["fp"] = int(fp)
    metrics["fn"] = int(fn)
    metrics["tp"] = int(tp)

    # Expected Calibration Error (ECE) — 10 bins
    metrics["ece"] = _compute_ece(y, y_prob, n_bins=10)

    log.info(
        "[%s] ROC-AUC=%.4f | PR-AUC=%.4f | Brier=%.4f | F1=%.4f | ECE=%.4f",
        model_name,
        metrics["roc_auc"],
        metrics["pr_auc"],
        metrics["brier_score"],
        metrics["f1"],
        metrics["ece"],
    )

    if save_plots:
        _plot_pr_curve(y, y_prob, model_name)
        _plot_calibration(y, y_prob, model_name)

    return metrics


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def find_optimal_threshold(y: pd.Series, y_prob: np.ndarray, metric: str = "f1") -> float:
    """Find probability threshold that maximizes F1 (or precision/recall).

    Parameters
    ----------
    y : pd.Series
        True labels.
    y_prob : np.ndarray
        Predicted probabilities.
    metric : str
        'f1', 'precision', or 'recall'.

    Returns
    -------
    float
        Optimal threshold.
    """
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)

    if metric == "f1":
        idx = np.argmax(f1_scores[:-1])
    elif metric == "precision":
        idx = np.argmax(precisions[:-1])
    else:
        idx = np.argmax(recalls[:-1])

    optimal = float(thresholds[idx])
    log.info("Optimal threshold (%s): %.4f", metric, optimal)
    return optimal


def compare_models(
    results: dict[str, dict],
    save: bool = True,
) -> pd.DataFrame:
    """Create a comparison table of all model metrics.

    Parameters
    ----------
    results : dict
        Keys are model names, values are metric dicts from evaluate_model().

    Returns
    -------
    pd.DataFrame
        Summary table sorted by PR-AUC.
    """
    display_metrics = ["roc_auc", "pr_auc", "brier_score", "f1", "precision", "recall", "ece"]
    rows = []
    for name, m in results.items():
        rows.append({"model": name, **{k: m.get(k, np.nan) for k in display_metrics}})

    df = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)

    log.info("\n%s", df.to_string(index=False, float_format="%.4f"))

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
        log.info("Comparison table saved to %s", RESULTS_DIR / "model_comparison.csv")

    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

def _plot_pr_curve(y: pd.Series, y_prob: np.ndarray, model_name: str) -> None:
    """Save precision-recall curve plot."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    precision, recall, _ = precision_recall_curve(y, y_prob)
    auc = average_precision_score(y, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"PR-AUC = {auc:.4f}")
    ax.axhline(y.mean(), color="gray", linestyle="--", label=f"Baseline (prevalence = {y.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = FIGURES_DIR / f"pr_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("PR curve saved to %s", path)


def _plot_calibration(y: pd.Series, y_prob: np.ndarray, model_name: str) -> None:
    """Save calibration (reliability) diagram."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(prob_pred, prob_true, "s-", label=model_name, lw=2)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve — {model_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = FIGURES_DIR / f"calibration_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Calibration curve saved to %s", path)
