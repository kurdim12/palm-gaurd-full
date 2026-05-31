"""TreeVibes (real RPW) ingest adapter.

Turns the TreeVibes corpus into Palm Guard ManifestRows with the exact schema in
:data:`config.MANIFEST_COLUMNS`. Everything downstream stays dataset-agnostic.

TreeVibes is hosted on Kaggle (``potamitis/treevibes``), which is **not** a plain
HTTP download — it needs either the Kaggle API (credentials) or a file you have
already downloaded. The adapter therefore accepts, in priority order:

1. ``TREEVIBES_LOCAL`` — a path to an already-downloaded ``.zip`` archive *or* an
   already-extracted folder. The offline-friendly path; no network needed.
2. ``TREEVIBES_KAGGLE`` — a Kaggle dataset slug (e.g. ``potamitis/treevibes``)
   fetched via the ``kaggle`` package. Requires Kaggle credentials
   (``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``).
3. ``TREEVIBES_URL`` — a direct archive URL (rare; most mirrors need auth).

The archive layout differs across releases; :data:`LABEL_DIR_HINTS` maps
folder-name fragments to our two classes. Adjust the hints (not downstream code)
if a release uses different folder names. The **site** (tree) is inferred from the
recording folder so clips from one tree never span the train/test split.
"""

from __future__ import annotations

import os
import tarfile
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

#: Default Kaggle dataset slug for TreeVibes.
KAGGLE_SLUG = "potamitis/treevibes"


# --------------------------------------------------------------------------------------
# Source acquisition (local / Kaggle / URL) -> an extracted folder.
# --------------------------------------------------------------------------------------

def _download_url(url: str, dest: Path) -> Path:
    """Download a direct ``url`` to ``dest`` (skipped if already present)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    with urllib.request.urlopen(url) as resp, dest.open("wb") as fh:  # noqa: S310
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return dest


def _download_kaggle(slug: str, dest_dir: Path) -> Path:
    """Download + unzip a Kaggle dataset via the official API.

    Requires the ``kaggle`` package and credentials. Returns the folder Kaggle
    extracted into.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "The 'kaggle' package is required to fetch TreeVibes from Kaggle. "
            "Install it (`pip install kaggle`) and provide credentials, or set "
            "TREEVIBES_LOCAL to an already-downloaded archive/folder."
        ) from exc

    dest_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
    api.dataset_download_files(slug, path=str(dest_dir), unzip=True, quiet=False)
    return dest_dir


def _extract(archive: Path, dest: Path) -> Path:
    """Extract a .zip or .tar(.gz) archive (idempotent)."""
    dest.mkdir(parents=True, exist_ok=True)
    name = archive.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    elif name.endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(archive) as tf:
            tf.extractall(dest)  # noqa: S202 - trusted dataset archive
    else:
        raise ValueError(f"Unsupported archive type: {archive.name}.")
    return dest


def _resolve_source(
    url: str | None,
    kaggle: str | None,
    local: str | None,
    work_dir: Path,
) -> Path:
    """Return a folder containing the extracted TreeVibes audio.

    Tries local → Kaggle → URL. Raises with actionable guidance if none works.
    """
    local = local or os.environ.get("TREEVIBES_LOCAL", "")
    if local:
        src = Path(local).expanduser()
        if src.is_dir():
            return src
        if src.is_file():
            return _extract(src, work_dir / "extracted")
        raise RuntimeError(f"TREEVIBES_LOCAL={src} does not exist.")

    kaggle = kaggle or os.environ.get("TREEVIBES_KAGGLE", "")
    if kaggle:
        return _download_kaggle(kaggle, work_dir / "extracted")

    url = url or config.TREEVIBES_URL
    if url:
        archive = _download_url(url, work_dir / "treevibes.zip")
        return _extract(archive, work_dir / "extracted")

    raise RuntimeError(
        "No TreeVibes source configured. Set one of: TREEVIBES_LOCAL (path to a "
        "downloaded .zip or folder), TREEVIBES_KAGGLE (e.g. 'potamitis/treevibes' "
        "with Kaggle credentials), or TREEVIBES_URL (direct archive URL)."
    )


# --------------------------------------------------------------------------------------
# Indexing an extracted folder -> ManifestRows.
# --------------------------------------------------------------------------------------

def _label_for(path: Path) -> str | None:
    """Infer class from any ancestor folder name; None if undetermined."""
    for part in (p.lower() for p in path.parts):
        for hint, label in LABEL_DIR_HINTS.items():
            if hint in part:
                return label
    return None


def _site_for(path: Path, label: str) -> str:
    """Infer a stable site/tree id from the recording folder.

    Uses the immediate parent directory (the per-tree recording folder), prefixed
    with the label to keep ids unique across classes.
    """
    parent = path.parent.name or "unknown"
    return f"{label[:2]}-{parent}"


def index_folder(extracted: Path) -> list[ManifestRow]:
    """Walk an extracted TreeVibes folder into ManifestRows.

    Pure filesystem logic (no network), so it is unit-testable against a fixture.
    """
    rows: list[ManifestRow] = []
    for audio in sorted(extracted.rglob("*")):
        if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTS:
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
    return rows


def build_rows(
    url: str | None = None,
    kaggle: str | None = None,
    local: str | None = None,
    work_dir: Path | None = None,
) -> list[ManifestRow]:
    """Acquire (local/Kaggle/URL), extract, and index TreeVibes into ManifestRows.

    Raises:
        RuntimeError: if no source is configured or no labelled audio is found.
    """
    work_dir = work_dir or (config.PATHS.data_dir / "treevibes")
    extracted = _resolve_source(url, kaggle, local, work_dir)
    rows = index_folder(extracted)
    if not rows:
        raise RuntimeError(
            "No labelled audio found in the TreeVibes source. Check LABEL_DIR_HINTS "
            "against the archive's folder names, or that the path is correct."
        )
    return rows


def build_manifest(
    url: str | None = None,
    kaggle: str | None = None,
    local: str | None = None,
    work_dir: Path | None = None,
):
    """Build a manifest from TreeVibes alone."""
    return write_manifest(build_rows(url, kaggle, local, work_dir))
