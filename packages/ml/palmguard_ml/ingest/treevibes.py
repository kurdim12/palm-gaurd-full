"""TreeVibes (real RPW) ingest adapter.

Turns the TreeVibes archive into a Palm Guard manifest with the exact schema in
:data:`config.MANIFEST_COLUMNS`. Everything downstream stays dataset-agnostic.

Enable by setting ``TREEVIBES_URL`` (env or :data:`config.TREEVIBES_URL`). The
adapter downloads + extracts the archive, then walks it inferring the label and
**site** from the directory layout so the site-split holds (clips from one tree
never span train/test).

The archive layout differs across TreeVibes releases; :data:`LABEL_DIR_HINTS`
maps folder-name fragments to our two classes. Adjust the hints (not the
downstream code) if a new release uses different folder names.
"""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from .. import audio_io, config
from ..manifest import ManifestRow, write_manifest

#: Folder-name fragments → class. Lower-cased substring match.
LABEL_DIR_HINTS: dict[str, str] = {
    "infest": config.LABEL_INFESTED,
    "infected": config.LABEL_INFESTED,
    "positive": config.LABEL_INFESTED,
    "rpw": config.LABEL_INFESTED,
    "clean": config.LABEL_CLEAN,
    "healthy": config.LABEL_CLEAN,
    "negative": config.LABEL_CLEAN,
    "control": config.LABEL_CLEAN,
}

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}


def _download(url: str, dest: Path) -> Path:
    """Download ``url`` to ``dest`` (skipped if already present)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def _extract(archive: Path, dest: Path) -> Path:
    """Extract a zip archive (idempotent)."""
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        raise ValueError(f"Unsupported archive type: {archive.suffix}. Add a handler here.")
    return dest


def _label_for(path: Path) -> str | None:
    """Infer class from any ancestor folder name; None if undetermined."""
    parts = [p.lower() for p in path.parts]
    for part in parts:
        for hint, label in LABEL_DIR_HINTS.items():
            if hint in part:
                return label
    return None


def _site_for(path: Path, label: str) -> str:
    """Infer a stable site id from the recording folder.

    Uses the immediate parent directory (the per-tree recording folder). Prefixed
    with the label to keep ids unique across classes.
    """
    parent = path.parent.name or "unknown"
    return f"{label[:2]}-{parent}"


def build_rows(url: str | None = None, work_dir: Path | None = None) -> list[ManifestRow]:
    """Download, extract, and index TreeVibes into ManifestRows.

    Args:
        url: Override for :data:`config.TREEVIBES_URL`.
        work_dir: Where to download/extract (default: ``data/treevibes``).

    Raises:
        RuntimeError: if no URL is configured or no labelled audio is found.
    """
    url = url or config.TREEVIBES_URL
    if not url:
        raise RuntimeError(
            "TREEVIBES_URL is not set. Set it in the environment or config.py to ingest "
            "real data; otherwise use the synthetic generator."
        )
    work_dir = work_dir or (config.PATHS.data_dir / "treevibes")
    archive = _download(url, work_dir / "treevibes.zip")
    extracted = _extract(archive, work_dir / "extracted")

    rows: list[ManifestRow] = []
    for audio in sorted(extracted.rglob("*")):
        if audio.suffix.lower() not in AUDIO_EXTS or not audio.is_file():
            continue
        label = _label_for(audio)
        if label is None:
            continue
        signal, sr = audio_io.read_wav(audio)
        rows.append(
            ManifestRow(
                path=str(audio.relative_to(config.PATHS.root))
                if config.PATHS.root in audio.parents
                else str(audio),
                label=label,
                source="treevibes",
                site=_site_for(audio, label),
                sample_rate=sr,
                duration=round(len(signal) / sr, 4),
            )
        )
    if not rows:
        raise RuntimeError(
            "No labelled audio found in the TreeVibes archive. Check LABEL_DIR_HINTS "
            "against the archive's folder names."
        )
    return rows


def build_manifest(url: str | None = None, work_dir: Path | None = None):
    """Build a manifest from TreeVibes alone."""
    return write_manifest(build_rows(url, work_dir))
