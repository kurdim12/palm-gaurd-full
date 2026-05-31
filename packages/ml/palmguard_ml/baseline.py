"""Classical RandomForest baseline.

A fast, CPU-only, dependency-light sanity check on the feature vector. Not the
production model — that's the CNN — but a cheap floor that must work before any
deep model is trusted.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from . import config, evaluate
from .dataset import load_splits
from .evaluate import Metrics


def build_model(seed: int = config.RANDOM_SEED) -> Pipeline:
    """Standardise → RandomForest, with class weighting for the minority class."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def train_and_eval() -> tuple[Pipeline, Metrics]:
    """Train the baseline on window features (site-split) and evaluate per file.

    Trains at the window level, then aggregates window scores to a file/tree
    decision (max-pool) for recall-first thresholding and reporting.

    Returns:
        ``(fitted_pipeline, metrics)``. Persists the model + metrics to artifacts.
    """
    train, test = load_splits(want_cnn=False)
    model = build_model()
    model.fit(train.X_vec, train.y)

    window_scores = model.predict_proba(test.X_vec)[:, 1]
    file_true, file_score = evaluate.aggregate_to_files(test.y, window_scores, test.files)
    threshold = evaluate.best_threshold_for_recall(file_true, file_score)
    metrics = evaluate.evaluate(file_true, file_score, threshold=threshold)

    config.PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.PATHS.baseline_model)
    payload = {"model": "baseline_rf", "level": "file", **metrics.to_dict()}
    config.PATHS.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model, metrics


def predict_proba(model: Pipeline, X_vec: np.ndarray) -> np.ndarray:
    """Infested-class probabilities for a feature matrix."""
    return model.predict_proba(X_vec)[:, 1]
