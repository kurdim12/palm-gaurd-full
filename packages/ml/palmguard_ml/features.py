"""Feature extraction for Palm Guard — the single source of truth.

Two representations are produced from the same preprocessed signal:

* :func:`feature_vector` — a compact, interpretable vector for the classical
  baseline and for cheap on-device sanity checks.
* :func:`cnn_input` — the log-mel "image" the CNN consumes.

**Mirror training in inference** (CLAUDE.md): the edge device imports these exact
functions. Never re-implement the feature math elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config, dsp

#: Ordered names of the scalar features in :func:`feature_vector`.
FEATURE_NAMES: tuple[str, ...] = (
    "rms",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "spectral_bandwidth_hz",
    "band_energy_ratio",
    "peak_band_ratio",
    "burst_rate_hz",
    "burst_energy_mean",
    "burst_energy_std",
    "flatness",
)


@dataclass(frozen=True)
class BurstStats:
    """Summary of detected feeding bursts in a clip."""

    rate_hz: float
    energy_mean: float
    energy_std: float
    count: int


def detect_bursts(signal: np.ndarray, sr: int = config.SAMPLE_RATE) -> BurstStats:
    """Detect short feeding transients via short-time energy thresholding.

    A burst is a contiguous run of frames whose energy exceeds an adaptive
    threshold, with run length inside the literature burst window
    (:data:`config.BURST_MIN_MS`–:data:`config.BURST_MAX_MS`).
    """
    frames = dsp.frame_signal(signal)
    energy = np.mean(frames**2, axis=1)
    if energy.size == 0:
        return BurstStats(0.0, 0.0, 0.0, 0)

    med = float(np.median(energy))
    mad = float(np.median(np.abs(energy - med))) + 1e-9
    threshold = med + 3.0 * mad
    active = energy > threshold

    frame_dur_s = config.HOP_LENGTH / sr
    min_frames = max(1, int((config.BURST_MIN_MS / 1000.0) / frame_dur_s))
    max_frames = max(min_frames, int((config.BURST_MAX_MS / 1000.0) / frame_dur_s) + 1)

    bursts: list[float] = []
    run_start: int | None = None
    for i, on in enumerate(active):
        if on and run_start is None:
            run_start = i
        elif not on and run_start is not None:
            length = i - run_start
            if min_frames <= length <= max_frames:
                bursts.append(float(np.sum(energy[run_start:i])))
            run_start = None
    if run_start is not None:
        length = len(active) - run_start
        if min_frames <= length <= max_frames:
            bursts.append(float(np.sum(energy[run_start:])))

    duration_s = signal.shape[0] / sr
    rate = len(bursts) / duration_s if duration_s > 0 else 0.0
    if bursts:
        return BurstStats(rate, float(np.mean(bursts)), float(np.std(bursts)), len(bursts))
    return BurstStats(rate, 0.0, 0.0, 0)


def _spectral_shape(power: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (centroid_hz, bandwidth_hz, flatness) averaged over frames."""
    n_bins = power.shape[1]
    freqs = np.linspace(0, sr / 2, n_bins)
    mag = power.mean(axis=0) + 1e-12
    centroid = float(np.sum(freqs * mag) / np.sum(mag))
    bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * mag) / np.sum(mag)))
    gmean = float(np.exp(np.mean(np.log(mag))))
    amean = float(np.mean(mag))
    flatness = gmean / amean if amean > 0 else 0.0
    return centroid, bandwidth, flatness


def feature_vector(window: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Compute the scalar feature vector for one preprocessed window.

    Args:
        window: One window from :func:`dsp.windows` (mono, banded, fixed-length).
        sr: Sample rate (defaults to the fixed working rate).

    Returns:
        float32 array aligned with :data:`FEATURE_NAMES`.
    """
    sig = np.asarray(window, dtype=np.float32)
    rms = float(np.sqrt(np.mean(sig**2) + 1e-12))
    zcr = float(np.mean(np.abs(np.diff(np.sign(sig))) > 0))

    power = dsp.power_spectrogram(sig)
    centroid, bandwidth, flatness = _spectral_shape(power, sr)

    n_bins = power.shape[1]
    freqs = np.linspace(0, sr / 2, n_bins)
    spectrum = power.mean(axis=0) + 1e-12
    total = float(np.sum(spectrum))
    band_mask = (freqs >= config.BAND_LOW_HZ) & (freqs <= config.BAND_HIGH_HZ)
    band_energy_ratio = float(np.sum(spectrum[band_mask]) / total)
    peak_mask = np.abs(freqs - config.PEAK_HZ) <= 250.0
    peak_band_ratio = float(np.sum(spectrum[peak_mask]) / total)

    bursts = detect_bursts(sig, sr)

    return np.array(
        [
            rms,
            zcr,
            centroid,
            bandwidth,
            band_energy_ratio,
            peak_band_ratio,
            bursts.rate_hz,
            bursts.energy_mean,
            bursts.energy_std,
            flatness,
        ],
        dtype=np.float32,
    )


def cnn_input(window: np.ndarray) -> np.ndarray:
    """Log-mel image of one window -> shape ``(n_mels, n_time_frames, 1)``."""
    log_mel = dsp.log_mel_spectrogram(window)
    return log_mel[..., np.newaxis].astype(np.float32)


def feature_vectors_from_audio(
    signal: np.ndarray, sr: int = config.SAMPLE_RATE
) -> np.ndarray:
    """Raw audio -> preprocess -> windows -> per-window feature vectors.

    Returns shape ``(n_windows, n_features)``.
    """
    pre = dsp.preprocess(signal, sr)
    return np.stack([feature_vector(w, sr) for w in dsp.windows(pre)])


def cnn_inputs_from_audio(
    signal: np.ndarray, sr: int = config.SAMPLE_RATE
) -> np.ndarray:
    """Raw audio -> preprocess -> windows -> per-window CNN inputs.

    Returns shape ``(n_windows, n_mels, n_time_frames, 1)``.
    """
    pre = dsp.preprocess(signal, sr)
    return np.stack([cnn_input(w) for w in dsp.windows(pre)])
