"""ESC-50 ingest adapter — hard negatives only.

ESC-50 is a labelled environmental-sound corpus. Per the authoritative
BUILD_SPEC §0.1(C) we use its **non-insect** classes (wind, rain, engines, …) as
hard negatives mapped to ``clean`` — they make the model robust to field noise
without ever being labelled infested.

Site id is the ESC-50 *category* so the site-split groups by sound type (a model
mustn't memorise "this noise class = clean" by seeing the same category in train
and test).
"""

from __future__ import annotations

import csv
import urllib.request
import zipfile
from pathlib import Path

from .. import audio_io, config
from ..manifest import ManifestRow, write_manifest

#: ESC-50 category we must NOT treat as a clean negative.
_INSECT_CATEGORY = "insects"


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def _extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    return dest


def _read_meta(extracted: Path) -> list[dict[str, str]]:
    """Find and parse ``meta/esc50.csv`` (filename,fold,target,category,...)."""
    meta = next(extracted.rglob("esc50.csv"), None)
    if meta is None:
        raise RuntimeError("esc50.csv not found in the ESC-50 archive.")
    with meta.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build_rows(url: str | None = None, work_dir: Path | None = None) -> list[ManifestRow]:
    """Return ``clean`` ManifestRows for ESC-50's non-insect classes.

    Raises:
        RuntimeError: if no URL is configured (caller should degrade gracefully).
    """
    url = url or config.ESC50_URL
    if not url:
        raise RuntimeError("ESC50_URL is not set.")
    work_dir = work_dir or (config.PATHS.data_dir / "esc50")
    archive = _download(url, work_dir / "esc50.zip")
    extracted = _extract(archive, work_dir / "extracted")

    audio_root = next((p for p in extracted.rglob("audio") if p.is_dir()), extracted)
    rows: list[ManifestRow] = []
    for entry in _read_meta(extracted):
        category = entry.get("category", "").strip().lower()
        if category == _INSECT_CATEGORY:
            continue  # don't mislabel insect sounds as clean
        wav = audio_root / entry["filename"]
        if not wav.exists():
            continue
        signal, sr = audio_io.read_wav(wav)
        rows.append(
            ManifestRow(
                path=str(wav),
                label=config.LABEL_CLEAN,
                source="esc50",
                site=f"esc50-{category}",
                sample_rate=sr,
                duration=round(len(signal) / sr, 4),
            )
        )
    return rows


def build_manifest(url: str | None = None):
    """Build a manifest from ESC-50 negatives alone (mainly for inspection)."""
    return write_manifest(build_rows(url))
