"""Minimal WAV read/write helpers (SciPy-backed, no heavy audio deps)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from . import config


def read_wav(path: str | Path, target_sr: int = config.SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Read a WAV file as mono float32, resampled to ``target_sr``.

    Returns:
        ``(signal, target_sr)``.
    """
    sr, data = wavfile.read(str(path))
    arr = np.asarray(data)
    if arr.dtype.kind in "iu":
        max_val = float(np.iinfo(arr.dtype).max)
        arr = arr.astype(np.float32) / max_val
    else:
        arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != target_sr:
        arr = resample_poly(arr, target_sr, sr).astype(np.float32)
    return arr.astype(np.float32), target_sr


def write_wav(path: str | Path, signal: np.ndarray, sr: int = config.SAMPLE_RATE) -> None:
    """Write a float32 [-1, 1] signal to a 16-bit PCM WAV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(signal, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    wavfile.write(str(path), sr, pcm)
