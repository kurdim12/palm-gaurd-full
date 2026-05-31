"""Guard the fixed scientific constants (CLAUDE.md golden rule #1)."""

from __future__ import annotations

from palmguard_ml import config


def test_fixed_acoustic_constants_unchanged():
    # Authoritative BUILD_SPEC constants. Intentional changes must be flagged +
    # justified in docs/BUILD_SPEC.md — not silently tuned.
    assert config.SAMPLE_RATE == 8_000
    assert config.BAND_LOW_HZ == 200
    assert config.BAND_HIGH_HZ == 2_500
    assert config.PEAK_HZ == 2_250
    assert config.WINDOW_SEC == 1.0
    assert config.HOP_SEC == 0.5
    assert config.N_FFT == 1_024
    assert config.HOP_LENGTH == 256
    assert config.N_MELS == 64
    assert config.BAND_LOW_HZ < config.PEAK_HZ < config.BAND_HIGH_HZ


def test_classes_and_positive_label():
    assert config.CLASSES == ("clean", "infested")
    assert config.POSITIVE_LABEL == "infested"
    assert config.LABEL_TO_INT["infested"] == 1


def test_recall_target_is_high():
    assert config.TARGET_INFESTED_RECALL >= 0.9


def test_time_frame_geometry_is_consistent():
    assert config.N_TIME_FRAMES > 0
    assert config.WINDOW_SAMPLES == int(config.SAMPLE_RATE * config.WINDOW_SEC)
    assert config.MEL_FMIN_HZ <= config.BAND_LOW_HZ
    assert config.MEL_FMAX_HZ >= config.BAND_HIGH_HZ
