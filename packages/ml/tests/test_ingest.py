"""Ingest-adapter unit tests (no network).

Covers the ESC-50 hard-negative logic: insect sounds are excluded, every other
category becomes a ``clean`` negative, and the site id is the category so the
site-split groups by sound type.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from palmguard_ml import audio_io, config
from palmguard_ml.ingest import esc50


def _make_fake_esc50(root: Path) -> Path:
    """Create a tiny ESC-50-shaped tree: audio/*.wav + meta/esc50.csv."""
    audio = root / "audio"
    meta = root / "meta"
    audio.mkdir(parents=True)
    meta.mkdir(parents=True)
    rows = [
        ("1-100032-A-0.wav", "0", "dog"),
        ("1-115545-A-48.wav", "48", "wind"),
        ("2-77945-A-7.wav", "7", "insects"),  # must be excluded
        ("3-118657-A-10.wav", "10", "rain"),
    ]
    for fname, _, _ in rows:
        audio_io.write_wav(audio / fname, np.zeros(config.SAMPLE_RATE, dtype=np.float32))
    with (meta / "esc50.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "fold", "target", "category"])
        for fname, target, category in rows:
            w.writerow([fname, "1", target, category])
    return root


def test_esc50_excludes_insects_and_maps_rest_to_clean(tmp_path, monkeypatch):
    extracted = _make_fake_esc50(tmp_path / "extracted")
    monkeypatch.setattr(esc50, "_download", lambda url, dest: dest)
    monkeypatch.setattr(esc50, "_extract", lambda archive, dest: extracted)

    rows = esc50.build_rows(url="http://example.invalid/esc50.zip", work_dir=tmp_path)

    assert len(rows) == 3  # insects excluded
    assert all(r.label == config.LABEL_CLEAN for r in rows)
    assert all(r.source == "esc50" for r in rows)
    assert "esc50-insects" not in {r.site for r in rows}
    assert {"esc50-dog", "esc50-wind", "esc50-rain"} == {r.site for r in rows}


def test_combined_falls_back_to_none_without_urls(monkeypatch):
    # With no real source URLs, the combined builder produces nothing (caller then
    # falls back to synthetic).
    from palmguard_ml import ingest

    monkeypatch.setattr(config, "TREEVIBES_URL", "")
    monkeypatch.setattr(config, "ESC50_URL", "")
    assert ingest.build_combined() is None
