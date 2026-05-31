"""CNN architectures (TensorFlow). Imported lazily so the rest of the package —
features, baseline, inference via TFLite — works without TensorFlow installed.

The backbone is selected by :data:`config.CNN_BACKBONE`. ``small_cnn`` is a
compact from-scratch net sized for the log-mel input; ``mobilenet`` wires a
transfer-learning backbone for when real data justifies it.
"""

from __future__ import annotations

from . import config


def build_model(backbone: str | None = None):
    """Build and compile a Keras model for the fixed log-mel input shape.

    Args:
        backbone: Override for :data:`config.CNN_BACKBONE`.

    Returns:
        A compiled ``tf.keras.Model`` emitting a single sigmoid (P(infested)).
    """
    import tensorflow as tf  # noqa: PLC0415 — heavy optional dep

    backbone = backbone or config.CNN_BACKBONE
    input_shape = (config.N_MELS, config.N_TIME_FRAMES, 1)
    inputs = tf.keras.Input(shape=input_shape, name="log_mel")

    if backbone == "mobilenet":
        # Transfer learning: tile mono log-mel to 3 channels, resize for the backbone.
        x = tf.keras.layers.Resizing(96, 96)(inputs)
        x = tf.keras.layers.Concatenate()([x, x, x])
        base = tf.keras.applications.MobileNetV2(
            input_shape=(96, 96, 3), include_top=False, weights=None, pooling="avg"
        )
        x = base(x)
    else:  # small_cnn (default)
        x = tf.keras.layers.BatchNormalization()(inputs)
        for filters in (16, 32, 64):
            x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D(2)(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="infested")(x)

    model = tf.keras.Model(inputs, outputs, name=f"palmguard_{backbone}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        # Recall-first: track recall + PR-AUC during training, not just accuracy.
        metrics=[
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model
