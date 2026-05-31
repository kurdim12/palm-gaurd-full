"""Ingest-adapter unit tests (no network).

Covers:
* ESC-50 hard-negative logic (insects excluded, rest → clean, site = category).
* TreeVibes CSV-driven, published-list labelling: folder→label, FOLDER→site,
  AUDIO cross-check accounting, unlisted-folder fallback, and zero silent drops.
* The combined builder's graceful fallback when nothing is configured.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np
import pytest

from palmguard_ml import audio_io, config
from palmguard_ml.ingest import esc50, treevibes

_SIG = None


def _sig():
    global _SIG
    if _SIG is None:
        _SIG = np.zeros(config.SAMPLE_RATE, dtype=np.float32)
    return _SIG


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
        audio_io.write_wav(audio / fname, _sig())
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
# TreeVibes (CSV-driven, published-list labelling — no network)
# --------------------------------------------------------------------------------------

def _make_fake_treevibes(
    root: Path,
    folders: dict[int, int],          # folder_number -> n_clips
    audio_overrides: dict[int, int] | None = None,  # folder -> AUDIO value in CSV
    include_csv_rows: set[int] | None = None,        # folders that get a CSV row
) -> Path:
    """Build a TreeVibes-shaped tree + an annotation CSV.

    Mirrors the real layout (field/field/train/<class>/folder_NN/clip.wav). The
    CSV carries the FOLDER/AUDIO columns the spec keys on; ``audio_overrides``
    lets a test force an AUDIO value (default: 1 for infested-listed, 0 for
    clean-listed, else from the list it belongs to).
    """
    audio_overrides = audio_overrides or {}
    include_csv_rows = include_csv_rows if include_csv_rows is not None else set(folders)

    for folder, n in folders.items():
        # Put clips under a class-named dir purely for realism; labelling ignores it.
        cls = "infested" if folder in treevibes.INFESTED_FOLDERS else "clean"
        for i in range(n):
            wav = root / "field" / "field" / "train" / cls / f"folder_{folder}" / f"rec_{i}.wav"
            audio_io.write_wav(wav, _sig())

    csv_path = root / "annotations.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["FOLDER", "IMEI", "GPS_LAT", "GPS_LONG", "VISUAL", "AUDIO",
                    "CONFIRMATION", "COMMENT"])
        for folder in sorted(folders):
            if folder not in include_csv_rows:
                continue
            if folder in audio_overrides:
                audio = audio_overrides[folder]
            elif folder in treevibes.INFESTED_FOLDERS:
                audio = 1
            elif folder in treevibes.CLEAN_FOLDERS:
                audio = 0
            else:
                audio = 0
            # IMEI reused across trees on purpose (must NOT become the site).
            w.writerow([folder, "IMEI_SHARED", "25.0", "49.0", "1", audio, "1", ""])
    return root


def test_treevibes_published_lists_label_and_site(tmp_path):
    # 2 infested folders + 2 clean folders, multiple clips each.
    root = _make_fake_treevibes(tmp_path / "tv", {1: 3, 2: 2, 7: 2, 10: 3})
    rows, report = treevibes.build_rows_with_report(
        local=str(root), work_dir=tmp_path / "work"
    )

    assert report.total_wavs == 10
    assert report.kept == 10
    # Zero unexpected drops.
    assert report.dropped_no_csv_row == 0
    assert report.dropped_unparsable_folder == 0
    assert report.dropped_unrecognized_audio == 0
    # Both classes present, by published list (folders 1,2 infested; 7,10 clean).
    assert report.by_label == {config.LABEL_INFESTED: 5, config.LABEL_CLEAN: 5}
    # SITE == FOLDER number (NOT the shared IMEI).
    assert {r.site for r in rows} == {"1", "2", "7", "10"}
    by_site_label = {r.site: r.label for r in rows}
    assert by_site_label["1"] == config.LABEL_INFESTED
    assert by_site_label["7"] == config.LABEL_CLEAN
    assert all(r.source == "treevibes" for r in rows)


def test_treevibes_audio_disagreement_is_counted_not_silent(tmp_path):
    # Folder 1 is on the INFESTED list but its AUDIO says 0 (clean) -> list wins,
    # disagreement counted.
    root = _make_fake_treevibes(
        tmp_path / "tv", {1: 2, 7: 2}, audio_overrides={1: 0}
    )
    rows, report = treevibes.build_rows_with_report(
        local=str(root), work_dir=tmp_path / "work"
    )
    assert report.audio_list_disagreements == 1
    # List membership wins: folder 1 stays infested despite AUDIO=0.
    assert {r.site: r.label for r in rows}["1"] == config.LABEL_INFESTED
    assert report.kept == 4


def test_treevibes_unlisted_folder_labelled_by_audio(tmp_path):
    # Folder 30 is in neither list (a test folder) -> label from AUDIO column.
    root = _make_fake_treevibes(
        tmp_path / "tv", {1: 1, 7: 1, 30: 2}, audio_overrides={30: 1}
    )
    rows, report = treevibes.build_rows_with_report(
        local=str(root), work_dir=tmp_path / "work"
    )
    assert report.unlisted_folders_labelled_by_audio == 1
    assert {r.site: r.label for r in rows}["30"] == config.LABEL_INFESTED
    assert report.kept == 4


def test_treevibes_missing_csv_row_is_counted_not_dropped_silently(tmp_path):
    # Folder 2 has clips on disk but NO CSV row -> its clips are counted as drops.
    root = _make_fake_treevibes(
        tmp_path / "tv", {1: 2, 2: 3, 7: 2}, include_csv_rows={1, 7}
    )
    rows, report = treevibes.build_rows_with_report(
        local=str(root), work_dir=tmp_path / "work"
    )
    assert report.total_wavs == 7
    assert report.dropped_no_csv_row == 3      # folder_2's 3 clips
    assert report.kept == 4
    assert "2" not in {r.site for r in rows}


def test_treevibes_local_zip_is_extracted(tmp_path):
    root = _make_fake_treevibes(tmp_path / "tv", {1: 2, 7: 2})
    archive = tmp_path / "treevibes.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for f in root.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(root.parent))

    rows = treevibes.build_rows(local=str(archive), work_dir=tmp_path / "work")
    assert len(rows) == 4
    assert {r.label for r in rows} == {config.LABEL_INFESTED, config.LABEL_CLEAN}


def test_treevibes_missing_local_raises(tmp_path):
    with pytest.raises(RuntimeError):
        treevibes.build_rows(local=str(tmp_path / "nope.zip"), work_dir=tmp_path)


def test_treevibes_no_source_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "TREEVIBES_LOCAL", "")
    monkeypatch.setattr(config, "TREEVIBES_KAGGLE", "")
    monkeypatch.setattr(config, "TREEVIBES_URL", "")
    with pytest.raises(RuntimeError):
        treevibes.build_rows(work_dir=tmp_path)


def test_treevibes_no_csv_raises(tmp_path):
    # Audio present but no annotation CSV anywhere -> explicit error, not a guess.
    root = tmp_path / "tv"
    audio_io.write_wav(root / "field" / "folder_1" / "a.wav", _sig())
    with pytest.raises(RuntimeError):
        treevibes.build_rows(local=str(root), work_dir=tmp_path / "work")


# --------------------------------------------------------------------------------------
# Combined builder
# --------------------------------------------------------------------------------------

def test_combined_falls_back_to_none_without_sources(monkeypatch):
    from palmguard_ml import ingest

    monkeypatch.setattr(config, "TREEVIBES_LOCAL", "")
    monkeypatch.setattr(config, "TREEVIBES_KAGGLE", "")
    monkeypatch.setattr(config, "TREEVIBES_URL", "")
    monkeypatch.setattr(config, "ESC50_URL", "")
    assert ingest.build_combined() is None


def test_combined_includes_treevibes_local(tmp_path, monkeypatch):
    root = _make_fake_treevibes(tmp_path / "tv", {1: 2, 2: 2, 7: 2, 10: 2})
    monkeypatch.setattr(config, "PATHS", config.Paths(root=tmp_path))
    monkeypatch.setattr(config, "TREEVIBES_LOCAL", str(root))
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
    assert {r.label for r in rows} == {config.LABEL_INFESTED, config.LABEL_CLEAN}
