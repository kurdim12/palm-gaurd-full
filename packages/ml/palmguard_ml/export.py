"""Quantise the Keras CNN to TFLite and verify parity.

Edge-first (CLAUDE.md #4): the on-device artifact is a quantised TFLite model.
Quantisation can shift outputs, so we assert the TFLite engine agrees with the
float Keras engine on a representative sample before trusting it.
"""

from __future__ import annotations

import json

import numpy as np

from . import config
from .dataset import load_splits


def _representative_dataset(x_cnn: np.ndarray):
    """Yield calibration samples for full-integer quantisation."""

    def gen():
        for i in range(min(len(x_cnn), 200)):
            yield [x_cnn[i : i + 1].astype(np.float32)]

    return gen


def export_tflite(quantize: bool = True):
    """Convert the saved Keras model to TFLite (optionally int8).

    Returns:
        Path to the written ``.tflite`` artifact.
    """
    import tensorflow as tf  # noqa: PLC0415

    model = tf.keras.models.load_model(config.PATHS.keras_model)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        train_ds, _ = load_splits(want_cnn=True)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = _representative_dataset(train_ds.X_cnn)

    tflite_bytes = converter.convert()
    config.PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.PATHS.tflite_model.write_bytes(tflite_bytes)
    return config.PATHS.tflite_model


def parity_check(tolerance: float = 0.05) -> dict:
    """Compare TFLite vs Keras P(infested) on the test split.

    Args:
        tolerance: max acceptable mean absolute probability difference.

    Returns:
        A report dict; raises if mean abs diff exceeds ``tolerance``.
    """
    from .inference import KerasEngine, TFLiteEngine

    _, test_ds = load_splits(want_cnn=True)
    keras = KerasEngine()
    tfl = TFLiteEngine()

    # Compare directly on the pre-built CNN inputs so we isolate quantisation
    # error from the (identical) feature path.
    diffs = []
    for i in range(len(test_ds)):
        batch = test_ds.X_cnn[i : i + 1]
        kp = float(keras._model.predict(batch, verbose=0)[0, 0])
        tp = _tflite_prob(tfl, batch)
        diffs.append(abs(kp - tp))

    mean_abs = float(np.mean(diffs)) if diffs else 0.0
    max_abs = float(np.max(diffs)) if diffs else 0.0
    report = {"mean_abs_diff": mean_abs, "max_abs_diff": max_abs, "tolerance": tolerance,
              "n": len(diffs), "passed": mean_abs <= tolerance}
    if not report["passed"]:
        raise AssertionError(f"TFLite parity failed: {report}")
    return report


def _tflite_prob(engine, x_cnn_batch: np.ndarray) -> float:
    """Run a single pre-built CNN input through the TFLite interpreter."""
    x = x_cnn_batch.astype(np.float32)
    in_d, out_d = engine._in, engine._out
    scale, zero = in_d.get("quantization", (0.0, 0))
    if in_d["dtype"] in (np.int8, np.uint8) and scale:
        x = np.round(x / scale + zero).astype(in_d["dtype"])
    engine._interpreter.set_tensor(in_d["index"], x)
    engine._interpreter.invoke()
    y = engine._interpreter.get_tensor(out_d["index"])
    out_scale, out_zero = out_d.get("quantization", (0.0, 0))
    if out_d["dtype"] in (np.int8, np.uint8) and out_scale:
        y = (y.astype(np.float32) - out_zero) * out_scale
    return float(np.asarray(y).reshape(-1)[0])


def export_and_verify() -> dict:
    """Full export path: convert, then parity-check. Returns the parity report."""
    export_tflite(quantize=True)
    report = parity_check()
    config.PATHS.metrics.parent.mkdir(parents=True, exist_ok=True)
    (config.PATHS.artifacts_dir / "parity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report
