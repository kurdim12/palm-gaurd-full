"""Inference interface — the edge/cloud boundary.

CLAUDE.md golden rule #4: inference must run **on-device** (TFLite) with no
connectivity. All inference goes through :class:`InferenceEngine`, so the edge,
the tests, and any cloud retraining job share one contract.

Three engines are provided:

* :class:`TFLiteEngine`    — production, runs the quantised model on the Pi.
* :class:`BaselineEngine`  — the classical RandomForest (CPU, no TF).
* :class:`KerasEngine`     — the float Keras model, for parity checks.

The feature path is always :mod:`palmguard_ml.features` (mirror training in
inference — never re-implement).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, dsp, features


@dataclass(frozen=True)
class Prediction:
    """One clip's classification result."""

    label: str
    confidence: float          # probability of the predicted label
    infested_prob: float       # probability of the infested class
    threshold: float

    @property
    def is_infested(self) -> bool:
        return self.label == config.LABEL_INFESTED


class InferenceEngine(ABC):
    """Common interface for all classifiers."""

    def __init__(self, threshold: float = config.DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold

    @abstractmethod
    def infested_prob(self, window: np.ndarray) -> float:
        """Return P(infested) for a single preprocessed 1 s window."""

    def predict(self, signal: np.ndarray, sr: int = config.SAMPLE_RATE) -> Prediction:
        """Classify raw audio: preprocess → window → aggregate (max-pool).

        Recall-first aggregation: the file is as infested as its most infested
        window, mirroring training-time file aggregation.
        """
        pre = dsp.preprocess(signal, sr)
        probs = [float(self.infested_prob(w)) for w in dsp.windows(pre)]
        return self._decide(max(probs) if probs else 0.0)

    def predict_window(self, window: np.ndarray) -> Prediction:
        """Classify a single already-preprocessed window."""
        return self._decide(float(self.infested_prob(window)))

    def _decide(self, p: float) -> Prediction:
        if p >= self.threshold:
            return Prediction(config.LABEL_INFESTED, p, p, self.threshold)
        return Prediction(config.LABEL_CLEAN, 1.0 - p, p, self.threshold)


class BaselineEngine(InferenceEngine):
    """RandomForest baseline engine (loads ``baseline.joblib``)."""

    def __init__(self, model_path: Path | None = None, threshold: float = config.DEFAULT_THRESHOLD):
        super().__init__(threshold)
        import joblib

        self._model = joblib.load(model_path or config.PATHS.baseline_model)

    def infested_prob(self, preprocessed: np.ndarray) -> float:
        vec = features.feature_vector(preprocessed)[np.newaxis, :]
        return float(self._model.predict_proba(vec)[0, 1])


class KerasEngine(InferenceEngine):
    """Float Keras CNN engine (for parity reference; needs TensorFlow)."""

    def __init__(self, model_path: Path | None = None, threshold: float = config.DEFAULT_THRESHOLD):
        super().__init__(threshold)
        import tensorflow as tf  # noqa: PLC0415 — heavy optional dep

        self._model = tf.keras.models.load_model(model_path or config.PATHS.keras_model)

    def infested_prob(self, preprocessed: np.ndarray) -> float:
        x = features.cnn_input(preprocessed)[np.newaxis, ...]
        return float(self._model.predict(x, verbose=0)[0, 0])


class TFLiteEngine(InferenceEngine):
    """Quantised TFLite engine — the on-device production path.

    Uses ``tflite_runtime`` if available (the Pi), else falls back to TensorFlow's
    bundled interpreter (dev machines).
    """

    def __init__(self, model_path: Path | None = None, threshold: float = config.DEFAULT_THRESHOLD):
        super().__init__(threshold)
        path = str(model_path or config.PATHS.tflite_model)
        self._interpreter = self._make_interpreter(path)
        self._interpreter.allocate_tensors()
        self._in = self._interpreter.get_input_details()[0]
        self._out = self._interpreter.get_output_details()[0]

    @staticmethod
    def _make_interpreter(path: str):
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            from tensorflow.lite import Interpreter  # noqa: PLC0415
        return Interpreter(model_path=path)

    def infested_prob(self, preprocessed: np.ndarray) -> float:
        x = features.cnn_input(preprocessed)[np.newaxis, ...].astype(np.float32)
        scale, zero = self._in.get("quantization", (0.0, 0))
        if self._in["dtype"] in (np.int8, np.uint8) and scale:
            x = np.round(x / scale + zero).astype(self._in["dtype"])
        self._interpreter.set_tensor(self._in["index"], x)
        self._interpreter.invoke()
        y = self._interpreter.get_tensor(self._out["index"])
        out_scale, out_zero = self._out.get("quantization", (0.0, 0))
        if self._out["dtype"] in (np.int8, np.uint8) and out_scale:
            y = (y.astype(np.float32) - out_zero) * out_scale
        return float(np.asarray(y).reshape(-1)[0])


def load_default_engine(threshold: float = config.DEFAULT_THRESHOLD) -> InferenceEngine:
    """Best available engine: TFLite if present, else baseline.

    Keeps the edge offline-capable: never reaches for the cloud.
    """
    if config.PATHS.tflite_model.exists():
        return TFLiteEngine(threshold=threshold)
    if config.PATHS.baseline_model.exists():
        return BaselineEngine(threshold=threshold)
    raise FileNotFoundError(
        "No model artifact found. Run `make baseline` or `make export` first."
    )
