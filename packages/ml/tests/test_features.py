"""Feature extraction + DSP shape/behaviour tests (window-based pipeline)."""

from __future__ import annotations

import numpy as np

from palmguard_ml import config, dsp, features


def test_windows_have_fixed_length(infested_clip):
    pre = dsp.preprocess(infested_clip)
    wins = dsp.windows(pre)
    assert wins.ndim == 2
    assert wins.shape[1] == config.WINDOW_SAMPLES
    assert wins.shape[0] >= 2  # a 3 s clip yields several 1 s / 0.5 s windows


def test_short_signal_yields_one_padded_window():
    # Shorter than one 1 s window (0.3 s) -> a single zero-padded window.
    short = (np.random.default_rng(0).standard_normal(int(0.3 * config.SAMPLE_RATE))
             .astype(np.float32) * 0.1)
    wins = dsp.windows(dsp.preprocess(short))
    assert wins.shape == (1, config.WINDOW_SAMPLES)


def test_feature_vector_shape_and_finiteness(clean_window):
    vec = features.feature_vector(clean_window)
    assert vec.shape == (len(features.FEATURE_NAMES),)
    assert np.all(np.isfinite(vec))


def test_cnn_input_shape(clean_window):
    img = features.cnn_input(clean_window)
    assert img.shape == (config.N_MELS, config.N_TIME_FRAMES, 1)
    assert img.dtype == np.float32


def test_feature_vectors_from_audio_is_per_window(infested_clip):
    mat = features.feature_vectors_from_audio(infested_clip)
    n_windows = dsp.windows(dsp.preprocess(infested_clip)).shape[0]
    assert mat.shape == (n_windows, len(features.FEATURE_NAMES))


def test_cnn_inputs_from_audio_is_per_window(infested_clip):
    arr = features.cnn_inputs_from_audio(infested_clip)
    n_windows = dsp.windows(dsp.preprocess(infested_clip)).shape[0]
    assert arr.shape == (n_windows, config.N_MELS, config.N_TIME_FRAMES, 1)


def test_infested_has_more_bursts_than_clean(clean_clip, infested_clip):
    clean_pre = dsp.windows(dsp.preprocess(clean_clip))[0]
    inf_pre = dsp.windows(dsp.preprocess(infested_clip))[0]
    assert features.detect_bursts(inf_pre).count > features.detect_bursts(clean_pre).count


def test_infested_concentrates_energy_near_peak(clean_clip, infested_clip):
    clean_vec = features.feature_vectors_from_audio(clean_clip).mean(axis=0)
    inf_vec = features.feature_vectors_from_audio(infested_clip).mean(axis=0)
    peak_idx = features.FEATURE_NAMES.index("peak_band_ratio")
    assert inf_vec[peak_idx] > clean_vec[peak_idx]


def test_bandpass_attenuates_out_of_band_tone():
    sr = config.SAMPLE_RATE
    n = sr  # 1 s
    t = np.arange(n) / sr
    # 3500 Hz: above the 2500 Hz band edge but below Nyquist (4000 Hz).
    out_of_band = np.sin(2 * np.pi * 3500 * t).astype(np.float32)
    filtered = dsp.bandpass(out_of_band)
    assert np.sqrt(np.mean(filtered**2)) < 0.5 * np.sqrt(np.mean(out_of_band**2))
