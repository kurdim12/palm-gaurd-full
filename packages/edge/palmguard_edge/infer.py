"""On-device inference wrapper.

Thin adapter over :class:`palmguard_ml.inference.InferenceEngine` so the edge runs
the same model + feature path as training. Prefers the quantised TFLite artifact;
falls back to the classical baseline if no TFLite model is present, so the device
is never left unable to classify offline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from palmguard_ml import config
from palmguard_ml.inference import (
    BaselineEngine,
    InferenceEngine,
    Prediction,
    TFLiteEngine,
)


def load_engine(model_path: str | Path, threshold: float = config.DEFAULT_THRESHOLD) -> InferenceEngine:
    """Load the best available engine for on-device inference."""
    path = Path(model_path)
    if path.exists() and path.suffix == ".tflite":
        return TFLiteEngine(model_path=path, threshold=threshold)
    baseline = config.PATHS.baseline_model
    if baseline.exists():
        return BaselineEngine(model_path=baseline, threshold=threshold)
    raise FileNotFoundError(
        f"No model at {path} and no baseline fallback at {baseline}. "
        "Run `make export` (or `make baseline`) first."
    )


def classify(engine: InferenceEngine, signal: np.ndarray) -> Prediction:
    """Classify a captured clip."""
    return engine.predict(signal)
