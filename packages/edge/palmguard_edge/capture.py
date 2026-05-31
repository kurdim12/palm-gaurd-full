"""Audio capture from the contact sensor.

On a Raspberry Pi this records from the ADC/sound card at the fixed
:data:`palmguard_ml.config.SAMPLE_RATE` (8 kHz). ``sounddevice`` is imported
lazily so the rest of the edge agent (and the test suite) runs on machines
without audio hardware. ``--sim`` mode reads a WAV instead of recording.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from palmguard_ml import audio_io, config


def record(duration_s: float = config.CLIP_DURATION_S) -> np.ndarray:
    """Record ``duration_s`` of mono audio at the fixed sample rate.

    Raises:
        RuntimeError: if no audio backend is available (e.g. dev machine).
    """
    try:
        import sounddevice as sd  # noqa: PLC0415 — hardware-only dep
    except ImportError as exc:  # pragma: no cover - hardware path
        raise RuntimeError(
            "sounddevice not installed; install it on the Pi or use --sim."
        ) from exc

    frames = int(duration_s * config.SAMPLE_RATE)
    audio = sd.rec(frames, samplerate=config.SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio.reshape(-1).astype(np.float32)


def load_clip(path: str | Path) -> np.ndarray:
    """Load a WAV clip (for ``--sim``), resampled to the fixed rate."""
    signal, _ = audio_io.read_wav(path)
    return signal
