"""Digital signal processing primitives for Palm Guard.

Pure-NumPy/SciPy DSP shared by training and edge inference. All functions are
stateless and operate on mono float32 signals at :data:`config.SAMPLE_RATE`.
Keep this re-implementation-free: the edge device imports these exact functions
so the feature path is bit-for-bit the same as training.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import config


def to_mono(signal: np.ndarray) -> np.ndarray:
    """Collapse a possibly multi-channel signal to mono float32."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1 if arr.shape[1] < arr.shape[0] else 0)
    return arr.astype(np.float32, copy=False)


def normalize_peak(signal: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Peak-normalise to [-1, 1]. Silent clips are returned unchanged."""
    arr = np.asarray(signal, dtype=np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak < eps:
        return arr
    return (arr / peak).astype(np.float32)


def fix_length(signal: np.ndarray, n_samples: int | None = None) -> np.ndarray:
    """Crop or zero-pad a signal to exactly ``n_samples`` (default: one clip)."""
    if n_samples is None:
        n_samples = int(config.SAMPLE_RATE * config.CLIP_DURATION_S)
    arr = np.asarray(signal, dtype=np.float32)
    if arr.shape[0] >= n_samples:
        return arr[:n_samples]
    out = np.zeros(n_samples, dtype=np.float32)
    out[: arr.shape[0]] = arr
    return out


def bandpass(signal: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Zero-phase Butterworth bandpass over the fixed RPW band (200–2500 Hz).

    Uses the fixed :data:`config.BAND_LOW_HZ` / :data:`config.BAND_HIGH_HZ`.
    """
    nyq = 0.5 * sr
    low = config.BAND_LOW_HZ / nyq
    high = min(config.BAND_HIGH_HZ / nyq, 0.999)
    sos = butter(N=4, Wn=[low, high], btype="bandpass", output="sos")
    arr = np.asarray(signal, dtype=np.float64)
    if arr.shape[0] <= 12:  # too short for filtfilt padding
        return arr.astype(np.float32)
    return sosfiltfilt(sos, arr).astype(np.float32)


def preprocess(signal: np.ndarray, sr: int = config.SAMPLE_RATE) -> np.ndarray:
    """Canonical pre-feature pipeline: mono → bandpass → peak-norm → fixed length."""
    arr = to_mono(signal)
    arr = bandpass(arr, sr)
    arr = normalize_peak(arr)
    arr = fix_length(arr)
    return arr


def frame_signal(
    signal: np.ndarray,
    frame_length: int = config.FRAME_LENGTH,
    hop_length: int = config.HOP_LENGTH,
) -> np.ndarray:
    """Split into overlapping frames -> shape ``(n_frames, frame_length)``."""
    arr = np.asarray(signal, dtype=np.float32)
    if arr.shape[0] < frame_length:
        arr = fix_length(arr, frame_length)
    n_frames = 1 + (arr.shape[0] - frame_length) // hop_length
    idx = np.arange(frame_length)[None, :] + hop_length * np.arange(n_frames)[:, None]
    return arr[idx]


def power_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Magnitude-squared STFT -> shape ``(n_frames, n_fft // 2 + 1)``."""
    frames = frame_signal(signal)
    window = np.hanning(config.FRAME_LENGTH).astype(np.float32)
    spectra = np.fft.rfft(frames * window, n=config.N_FFT, axis=1)
    return (np.abs(spectra) ** 2).astype(np.float32)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr: int = config.SAMPLE_RATE,
    n_fft: int = config.N_FFT,
    n_mels: int = config.N_MELS,
    fmin: int = config.MEL_FMIN_HZ,
    fmax: int = config.MEL_FMAX_HZ,
) -> np.ndarray:
    """Triangular mel filterbank -> shape ``(n_mels, n_fft // 2 + 1)``.

    Bounded to the fixed RPW band so mel bins concentrate where the signal lives.
    """
    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2, n_bins)
    mel_min, mel_max = _hz_to_mel(np.array([fmin])), _hz_to_mel(np.array([fmax]))
    mel_points = np.linspace(mel_min[0], mel_max[0], n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.searchsorted(fft_freqs, hz_points)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        for k in range(n_bins):
            f = fft_freqs[k]
            if left <= f <= center and center > left:
                fb[m - 1, k] = (f - left) / (center - left)
            elif center <= f <= right and right > center:
                fb[m - 1, k] = (right - f) / (right - center)
    _ = bins  # documented intent; triangles computed directly above
    return fb


_MEL_FB = mel_filterbank()


def log_mel_spectrogram(signal: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram for the CNN -> shape ``(n_mels, n_time_frames)``.

    Operates on an *already preprocessed* signal (call :func:`preprocess` first).
    Padded/cropped to the fixed :data:`config.N_TIME_FRAMES`.
    """
    power = power_spectrogram(signal)            # (frames, bins)
    mel = power @ _MEL_FB.T                       # (frames, n_mels)
    log_mel = np.log(mel + 1e-6).astype(np.float32)
    log_mel = log_mel.T                           # (n_mels, frames)

    target = config.N_TIME_FRAMES
    if log_mel.shape[1] >= target:
        log_mel = log_mel[:, :target]
    else:
        pad = np.full((log_mel.shape[0], target - log_mel.shape[1]),
                      log_mel.min(), dtype=np.float32)
        log_mel = np.concatenate([log_mel, pad], axis=1)
    return log_mel
