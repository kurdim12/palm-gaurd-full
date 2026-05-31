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
    """Train the baseline on the site-split train set and evaluate on test.

    Returns:
        ``(fitted_pipeline, metrics)``. Persists the model + metrics to artifacts.
    """
    train, test = load_splits(want_cnn=False)
    model = build_model()
    model.fit(train.X_vec, train.y)

    scores = model.predict_proba(test.X_vec)[:, 1]
    threshold = evaluate.best_threshold_for_recall(test.y, scores)
    metrics = evaluate.evaluate(test.y, scores, threshold=threshold)

    config.PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.PATHS.baseline_model)
    payload = {"model": "baseline_rf", **metrics.to_dict()}
    config.PATHS.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model, metrics


def predict_proba(model: Pipeline, X_vec: np.ndarray) -> np.ndarray:
    """Infested-class probabilities for a feature matrix."""
    return model.predict_proba(X_vec)[:, 1]
