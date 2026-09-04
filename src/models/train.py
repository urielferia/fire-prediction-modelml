"""
src/models/train.py — Train Logistic Regression, Random Forest, and LightGBM.

Uses RandomizedSearchCV for hyperparameter tuning.
All models are trained on the sampled training split (20:1 ratio).
All evaluations are on the full validation split.

Usage:
    python -m src.models.train --model all
    python -m src.models.train --model lgbm
"""

from __future__ import annotations

import argparse
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from src.config import (
    FEATURES_TRAIN,
    FEATURES_VAL,
    MODEL_LR,
    MODEL_RF,
    MODEL_LGBM,
    PREPROCESSOR,
    RANDOM_SEED,
)
from src.features.static import TARGET_COL, NON_FEATURE_COLS, CATEGORICAL_COLS

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

N_ITER_SEARCH = 8     # Fast hyperparameter search iterations per model
CV_FOLDS = 3          # Cross-validation folds
SCORING = "average_precision"  # PR-AUC in sklearn


# ── Feature selection ─────────────────────────────────────────────────────────

def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return list of feature columns (excludes target and id columns)."""
    exclude = set(NON_FEATURE_COLS + [TARGET_COL])
    return [c for c in df.columns if c not in exclude]


# ── Data preparation ──────────────────────────────────────────────────────────

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits from parquet."""
    train = pd.read_parquet(FEATURES_TRAIN)
    val = pd.read_parquet(FEATURES_VAL)
    return train, val


def prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Extract X (features) and y (target) from a DataFrame."""
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int)
    return X, y


# ── Logistic Regression ───────────────────────────────────────────────────────

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train Logistic Regression with RandomizedSearchCV."""
    log.info("Training Logistic Regression...")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            random_state=RANDOM_SEED,
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    param_dist = {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=4,
        scoring=SCORING,
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    log.info(
        "LR best params: %s | CV PR-AUC: %.4f",
        search.best_params_, search.best_score_,
    )
    return search.best_estimator_


# ── Random Forest ─────────────────────────────────────────────────────────────

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train Random Forest with RandomizedSearchCV."""
    log.info("Training Random Forest...")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_SEED,
            n_jobs=-1,
            class_weight="balanced",
        )),
    ])

    param_dist = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [15, 25],
        "clf__min_samples_leaf": [10, 50],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=SCORING,
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    log.info(
        "RF best params: %s | CV PR-AUC: %.4f",
        search.best_params_, search.best_score_,
    )
    return search.best_estimator_


# ── LightGBM ─────────────────────────────────────────────────────────────────

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> lgb.LGBMClassifier:
    """Train LightGBM with RandomizedSearchCV."""
    log.info("Training LightGBM...")

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    log.info("LightGBM scale_pos_weight: %.2f", scale_pos_weight)

    clf = lgb.LGBMClassifier(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        verbose=-1,
    )

    param_dist = {
        "n_estimators": [200, 500],
        "learning_rate": [0.03, 0.08],
        "max_depth": [7, 12],
        "num_leaves": [31, 63],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
    }

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        clf,
        param_dist,
        n_iter=N_ITER_SEARCH,
        scoring=SCORING,
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    log.info(
        "LGBM best params: %s | CV PR-AUC: %.4f",
        search.best_params_, search.best_score_,
    )
    return search.best_estimator_


# ── Persistence ───────────────────────────────────────────────────────────────

def save_model(model, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    log.info("Model saved to %s", path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(model_name: str = "all") -> None:
    log.info("Loading data splits...")
    train, val = load_splits()

    feature_cols = get_feature_cols(train)
    log.info("Feature count: %d", len(feature_cols))

    X_train, y_train = prepare_xy(train, feature_cols)
    X_val, y_val = prepare_xy(val, feature_cols)

    log.info(
        "Train: %d rows | %d positive (%.2f%%)",
        len(X_train), y_train.sum(), 100 * y_train.mean(),
    )

    joblib.dump({"feature_cols": feature_cols}, PREPROCESSOR)

    if model_name in ("all", "lr"):
        lr = train_logistic_regression(X_train, y_train)
        save_model(lr, MODEL_LR)

    if model_name in ("all", "rf"):
        rf = train_random_forest(X_train, y_train)
        save_model(rf, MODEL_RF)

    if model_name in ("all", "lgbm"):
        lgbm = train_lightgbm(X_train, y_train)
        save_model(lgbm, MODEL_LGBM)

    log.info("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wildfire risk models")
    parser.add_argument(
        "--model", choices=["all", "lr", "rf", "lgbm"], default="all",
        help="Which model to train (default: all)",
    )
    args = parser.parse_args()
    main(args.model)
