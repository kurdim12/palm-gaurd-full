"""Feature extraction + DSP shape/behaviour tests."""

from __future__ import annotations

import numpy as np

from palmguard_ml import config, dsp, features


def test_preprocess_fixed_length_and_mono(infested_clip):
    pre = dsp.preprocess(infested_clip)
    assert pre.ndim == 1
    assert pre.shape[0] == int(config.SAMPLE_RATE * config.CLIP_DURATION_S)
    assert np.max(np.abs(pre)) <= 1.0 + 1e-6


def test_feature_vector_shape_and_finiteness(clean_clip):
    vec = features.features_from_audio(clean_clip)
    assert vec.shape == (len(features.FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_cnn_input_shape(infested_clip):
    img = features.cnn_input_from_audio(infested_clip)
    assert img.shape == (config.N_MELS, config.N_TIME_FRAMES, 1)
    assert img.dtype == np.float32


def test_infested_has_more_bursts_than_clean(clean_clip, infested_clip):
    clean_bursts = features.detect_bursts(dsp.preprocess(clean_clip))
    infested_bursts = features.detect_bursts(dsp.preprocess(infested_clip))
    assert infested_bursts.count > clean_bursts.count


def test_infested_concentrates_energy_near_peak(clean_clip, infested_clip):
    clean_vec = features.features_from_audio(clean_clip)
    inf_vec = features.features_from_audio(infested_clip)
    peak_idx = features.FEATURE_NAMES.index("peak_band_ratio")
    assert inf_vec[peak_idx] > clean_vec[peak_idx]


def test_bandpass_attenuates_out_of_band_tone():
    sr = config.SAMPLE_RATE
    n = sr  # 1 s
    t = np.arange(n) / sr
    out_of_band = np.sin(2 * np.pi * 3500 * t).astype(np.float32)  # above 2.5 kHz
    filtered = dsp.bandpass(out_of_band)
    assert np.sqrt(np.mean(filtered**2)) < 0.5 * np.sqrt(np.mean(out_of_band**2))
