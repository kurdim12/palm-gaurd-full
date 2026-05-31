"""Ingest-adapter unit tests (no network).

Covers:
* ESC-50 hard-negative logic (insects excluded, rest → clean, site = category).
* TreeVibes local ingestion from a folder and from a .zip, label + site inference.
* The combined builder's graceful fallback when nothing is configured.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np

from palmguard_ml import audio_io, config
from palmguard_ml.ingest import esc50, treevibes


# --------------------------------------------------------------------------------------
# ESC-50
# --------------------------------------------------------------------------------------

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


# --------------------------------------------------------------------------------------
# TreeVibes (local path — no network)
# --------------------------------------------------------------------------------------

def _make_fake_treevibes(root: Path) -> Path:
    """A small TreeVibes-shaped tree: <class>/<tree>/clip.wav."""
    layout = {
        ("infested", "tree_A"): 2,
        ("infested", "tree_B"): 2,
        ("clean", "tree_C"): 2,
        ("healthy", "tree_D"): 2,  # 'healthy' must map to clean too
    }
    sig = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    for (cls, tree), n in layout.items():
        for i in range(n):
            audio_io.write_wav(root / cls / tree / f"rec_{i}.wav", sig)
    return root


def test_treevibes_local_folder_labels_and_sites(tmp_path):
    extracted = _make_fake_treevibes(tmp_path / "tv")
    rows = treevibes.build_rows(local=str(extracted), work_dir=tmp_path / "work")

    assert len(rows) == 8
    labels = {r.site: r.label for r in rows}
    # 'healthy' folder maps to clean via LABEL_DIR_HINTS.
    assert set(labels.values()) == {config.LABEL_INFESTED, config.LABEL_CLEAN}
    # Site ids are per-tree and label-prefixed, so no tree spans classes.
    assert {r.site for r in rows} == {"in-tree_A", "in-tree_B", "cl-tree_C", "cl-tree_D"}
    assert all(r.source == "treevibes" for r in rows)


def test_treevibes_real_layout_nesting_and_unlabeled(tmp_path):
    """Mirror the real TreeVibes tree: nested class dirs, per-tree subfolders that
    nest deeper, and an unlabeled test/ folder that must be skipped."""
    root = tmp_path / "tv"
    sig = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    paths = [
        "field/field/train/clean/folder_10/a.wav",
        "field/field/train/clean/folder_8/fig1clips_x/c.wav",   # deeper nesting
        "field/field/train/clean/folder_8/fig2_clips_y/d.wav",  # same tree, deeper
        "field/field/train/clean/folder_9/folder_9/e.wav",      # double-nested
        "field/field/train/infested/folder_1/f.wav",
        "field/field/train/infested/folder_2/g.wav",
        "field/field/test/folder_26/h.wav",          # unlabeled -> skipped
        "field/field/test/difficult_cases/i.wav",    # unlabeled -> skipped
    ]
    for p in paths:
        audio_io.write_wav(root / p, sig)

    rows = treevibes.build_rows(local=str(root), work_dir=tmp_path / "work")

    # The two test/ clips have no clean|infested ancestor -> skipped.
    assert len(rows) == 6
    # folder_8's two deeply-nested clips collapse to a single tree/site.
    f8 = {r.site for r in rows if "folder_8" in r.path}
    assert f8 == {"cl-folder_8"}
    # Site = the folder directly under the class dir (the tree), not deeper dirs.
    assert {r.site for r in rows} == {
        "cl-folder_10", "cl-folder_8", "cl-folder_9",
        "in-folder_1", "in-folder_2",
    }


def test_treevibes_local_zip_is_extracted(tmp_path):
    extracted = _make_fake_treevibes(tmp_path / "tv")
    archive = tmp_path / "treevibes.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for wav in extracted.rglob("*.wav"):
            zf.write(wav, wav.relative_to(extracted.parent))

    rows = treevibes.build_rows(local=str(archive), work_dir=tmp_path / "work")
    assert len(rows) == 8
    assert {r.label for r in rows} == {config.LABEL_INFESTED, config.LABEL_CLEAN}


def test_treevibes_missing_local_raises(tmp_path):
    import pytest

    with pytest.raises(RuntimeError):
        treevibes.build_rows(local=str(tmp_path / "nope.zip"), work_dir=tmp_path)


def test_treevibes_no_source_raises(tmp_path, monkeypatch):
    import pytest

    monkeypatch.setattr(config, "TREEVIBES_LOCAL", "")
    monkeypatch.setattr(config, "TREEVIBES_KAGGLE", "")
    monkeypatch.setattr(config, "TREEVIBES_URL", "")
    monkeypatch.delenv("TREEVIBES_LOCAL", raising=False)
    monkeypatch.delenv("TREEVIBES_KAGGLE", raising=False)
    with pytest.raises(RuntimeError):
        treevibes.build_rows(work_dir=tmp_path)


# --------------------------------------------------------------------------------------
# Combined builder
# --------------------------------------------------------------------------------------

def test_combined_falls_back_to_none_without_sources(monkeypatch):
    # With no real source configured, the combined builder produces nothing
    # (caller then falls back to synthetic).
    from palmguard_ml import ingest

    monkeypatch.setattr(config, "TREEVIBES_LOCAL", "")
    monkeypatch.setattr(config, "TREEVIBES_KAGGLE", "")
    monkeypatch.setattr(config, "TREEVIBES_URL", "")
    monkeypatch.setattr(config, "ESC50_URL", "")
    assert ingest.build_combined() is None


def test_combined_includes_treevibes_local(tmp_path, monkeypatch):
    extracted = _make_fake_treevibes(tmp_path / "tv")
    monkeypatch.setattr(config, "PATHS", config.Paths(root=tmp_path))
    monkeypatch.setattr(config, "TREEVIBES_LOCAL", str(extracted))
    monkeypatch.setattr(config, "TREEVIBES_KAGGLE", "")
    monkeypatch.setattr(config, "TREEVIBES_URL", "")
    monkeypatch.setattr(config, "ESC50_URL", "")

    from palmguard_ml import ingest
    from palmguard_ml.manifest import read_manifest

    manifest = ingest.build_combined()
    assert manifest is not None
    rows = read_manifest(manifest)
    assert len(rows) == 8
    assert {r.source for r in rows} == {"treevibes"}
