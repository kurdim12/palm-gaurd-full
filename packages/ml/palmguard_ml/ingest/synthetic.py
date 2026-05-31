"""Synthetic RPW-like dataset generator.

Produces a manifest + WAVs that *imitate* the structure described in the RPW
acoustics literature, so the whole pipeline (features → baseline → eval → export)
can be exercised with no external download. This is a sanity fixture, **not** a
substitute for real TreeVibes data.

Signal model
------------
* ``clean``    : band-limited ambient noise (trunk/environment), no bursts.
* ``infested`` : the same ambient floor plus periodic short feeding *bursts* —
  decaying tone packets centred near :data:`config.PEAK_HZ` with inter-burst
  gaps drawn from the literature interval range.

Sites are simulated trees; clips inherit their site so the site-split is
meaningful (per-site noise colour and burst statistics vary).
"""

from __future__ import annotations

import numpy as np

from .. import audio_io, config
from ..manifest import ManifestRow, write_manifest


def _ambient(rng: np.random.Generator, n: int, colour: float) -> np.ndarray:
    """Band-ish coloured noise as the environmental floor."""
    white = rng.standard_normal(n).astype(np.float32)
    # Simple one-pole low-pass to vary "colour" per site.
    out = np.empty_like(white)
    acc = 0.0
    for i in range(n):
        acc = colour * acc + (1 - colour) * white[i]
        out[i] = acc
    return out * 0.2


def _burst(rng: np.random.Generator, sr: int, peak_hz: float) -> np.ndarray:
    """One decaying tone-packet feeding transient."""
    dur_ms = rng.uniform(config.BURST_MIN_MS, config.BURST_MAX_MS)
    n = max(4, int(sr * dur_ms / 1000.0))
    t = np.arange(n) / sr
    freq = peak_hz * rng.uniform(0.9, 1.05)
    decay = np.exp(-t / (dur_ms / 3000.0))
    tone = np.sin(2 * np.pi * freq * t) * decay
    # A touch of broadband click at onset.
    tone[: max(1, n // 10)] += rng.standard_normal(max(1, n // 10)) * 0.5
    return (tone * rng.uniform(0.6, 1.0)).astype(np.float32)


def _make_clip(rng: np.random.Generator, infested: bool, colour: float) -> np.ndarray:
    sr = config.SAMPLE_RATE
    n = int(sr * config.CLIP_DURATION_S)
    sig = _ambient(rng, n, colour)
    if infested:
        pos = 0
        while pos < n:
            gap = rng.uniform(config.BURST_INTERVAL_MIN_S, config.BURST_INTERVAL_MAX_S)
            pos += int(gap * sr)
            if pos >= n:
                break
            burst = _burst(rng, sr, config.PEAK_HZ)
            end = min(n, pos + burst.shape[0])
            sig[pos:end] += burst[: end - pos]
            pos = end
    peak = float(np.max(np.abs(sig))) or 1.0
    return (sig / peak * 0.9).astype(np.float32)


def build_manifest(
    n_sites_per_class: int = 6,
    clips_per_site: int = 8,
    seed: int = config.RANDOM_SEED,
):
    """Generate synthetic audio + write the manifest.

    Args:
        n_sites_per_class: simulated trees per class.
        clips_per_site: clips recorded at each tree.
        seed: RNG seed for reproducibility.

    Returns:
        The path to the written manifest.
    """
    rng = np.random.default_rng(seed)
    raw_dir = config.PATHS.raw_dir
    rows: list[ManifestRow] = []

    for label, infested in ((config.LABEL_CLEAN, False), (config.LABEL_INFESTED, True)):
        for s in range(n_sites_per_class):
            site = f"{label[:2]}-site-{s:02d}"
            colour = float(rng.uniform(0.6, 0.95))
            site_dir = raw_dir / label / site
            for c in range(clips_per_site):
                clip = _make_clip(rng, infested, colour)
                wav_path = site_dir / f"clip_{c:03d}.wav"
                audio_io.write_wav(wav_path, clip)
                rows.append(
                    ManifestRow(
                        path=str(wav_path.relative_to(config.PATHS.root)),
                        label=label,
                        source="synthetic",
                        site=site,
                        sample_rate=config.SAMPLE_RATE,
                        duration=config.CLIP_DURATION_S,
                    )
                )

    return write_manifest(rows)
