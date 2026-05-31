"""Shared fixtures for the ML test suite.

Tests must never depend on `make data` having been run, so fixtures synthesise
small audio in-memory. The package is importable without installation via the
path shim below.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palmguard_ml import config  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _tone(freq: float, n: int, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


@pytest.fixture
def clean_clip(rng) -> np.ndarray:
    """Band-limited noise floor, no bursts (multi-window length)."""
    n = int(config.SAMPLE_RATE * config.SYNTH_CLIP_SEC)
    return (rng.standard_normal(n).astype(np.float32) * 0.1)


@pytest.fixture
def infested_clip(rng) -> np.ndarray:
    """Noise floor plus dense tone bursts near the RPW peak frequency."""
    sr = config.SAMPLE_RATE
    n = int(sr * config.SYNTH_CLIP_SEC)
    sig = rng.standard_normal(n).astype(np.float32) * 0.1
    burst_n = int(sr * 0.02)  # 20 ms bursts
    burst = 2.0 * _tone(config.PEAK_HZ, burst_n, sr) * np.exp(
        -np.arange(burst_n) / (burst_n / 3)
    )
    for start in range(0, n - burst_n, int(sr * 0.08)):  # every 80 ms
        sig[start : start + burst_n] += burst
    return sig


@pytest.fixture
def clean_window(rng) -> np.ndarray:
    """A single preprocessed-length clean window."""
    return rng.standard_normal(config.WINDOW_SAMPLES).astype(np.float32) * 0.1
