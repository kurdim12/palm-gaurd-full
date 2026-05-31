"""CNN training on the site-split, with minority-class handling.

Heavy (TensorFlow) and imported lazily by the CLI. Honours the same site-split as
the baseline so reported metrics are comparable and never leak a tree across
train/test.
"""

from __future__ import annotations

import json

import numpy as np

from . import config, evaluate
from .dataset import Dataset, load_splits


def _class_weights(y: np.ndarray) -> dict[int, float]:
    """Inverse-frequency weights so the costly infested class isn't ignored."""
    counts = np.bincount(y, minlength=2).astype(float)
    counts[counts == 0] = 1.0
    total = counts.sum()
    return {i: float(total / (2.0 * counts[i])) for i in range(2)}


def train(epochs: int = 30, batch_size: int = 32, backbone: str | None = None):
    """Train the CNN and persist the Keras model + metrics.

    Returns:
        ``(keras_model, metrics)``.
    """
    import tensorflow as tf  # noqa: PLC0415

    tf.keras.utils.set_random_seed(config.RANDOM_SEED)
    from .model import build_model

    train_ds, test_ds = load_splits(want_cnn=True)
    model = build_model(backbone)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True
        )
    ]
    model.fit(
        train_ds.X_cnn,
        train_ds.y,
        validation_data=(test_ds.X_cnn, test_ds.y),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=_class_weights(train_ds.y),
        callbacks=callbacks,
        verbose=2,
    )

    config.PATHS.artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(config.PATHS.keras_model)
    metrics = _evaluate(model, test_ds)
    payload = {"model": f"cnn_{backbone or config.CNN_BACKBONE}", **metrics.to_dict()}
    config.PATHS.metrics.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return model, metrics


def _evaluate(model, test_ds: Dataset):
    scores = model.predict(test_ds.X_cnn, verbose=0).reshape(-1)
    threshold = evaluate.best_threshold_for_recall(test_ds.y, scores)
    return evaluate.evaluate(test_ds.y, scores, threshold=threshold)


def evaluate_saved():
    """Reload the saved Keras model and re-evaluate on the test site-split."""
    import tensorflow as tf  # noqa: PLC0415

    model = tf.keras.models.load_model(config.PATHS.keras_model)
    _, test_ds = load_splits(want_cnn=True)
    return _evaluate(model, test_ds)
