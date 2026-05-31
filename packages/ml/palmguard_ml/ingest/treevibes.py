"""TreeVibes (real RPW) ingest adapter.

Turns the TreeVibes corpus into Palm Guard ManifestRows with the exact schema in
:data:`config.MANIFEST_COLUMNS`. Everything downstream stays dataset-agnostic.

TreeVibes is hosted on Kaggle (``potamitis/treevibes``), which is **not** a plain
HTTP download — it needs either the Kaggle API (credentials) or a file you have
already downloaded. The adapter therefore accepts, in priority order:

1. ``TREEVIBES_LOCAL`` — a path to an already-downloaded ``.zip`` archive *or* an
   already-extracted folder. The offline-friendly path; no network needed.
2. ``TREEVIBES_KAGGLE`` — a Kaggle dataset slug (e.g. ``potamitis/treevibes``)
   fetched via the ``kaggle`` package. Requires Kaggle credentials.
3. ``TREEVIBES_URL`` — a direct archive URL (rare; most mirrors need auth).

Labelling (authoritative spec)
-------------------------------
Labels are **folder-level**, taken from the published folder lists — *not* from
folder-name string matching and *not* from the CONFIRMATION column:

* INFESTED folders: 1,2,3,4,5,6,11,12,13,14,15,16,17,18,19,20,21,22,23
* CLEAN folders:    7,8,9,10,24,25,35

The annotation CSV has one row **per folder** with columns
``FOLDER, IMEI, GPS_LAT, GPS_LONG, VISUAL, AUDIO, CONFIRMATION, COMMENT``. For
folders in the published lists we cross-check the folder's ``AUDIO`` value
(1↔infested, 0↔clean) and **count** disagreements rather than silently choosing.
Folders not in either list (e.g. test folders 26–34) are labelled from their
``AUDIO`` column. ``SITE`` is the **FOLDER number** (one folder ≈ one tree) — not
the IMEI, since devices were reused across trees.

Every wav whose folder lacks a CSV row, or whose AUDIO value is unrecognised, is
**counted and reported** (see :class:`IngestReport`), never dropped silently.
"""

from __future__ import annotations

import csv
import re
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from .. import audio_io, config
from ..manifest import ManifestRow, write_manifest

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3"}

#: Default Kaggle dataset slug for TreeVibes.
KAGGLE_SLUG = "potamitis/treevibes"

#: Published folder lists (authoritative ground truth).
INFESTED_FOLDERS: frozenset[int] = frozenset(
    {1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
)
CLEAN_FOLDERS: frozenset[int] = frozenset({7, 8, 9, 10, 24, 25, 35})

#: AUDIO column value -> label (used for cross-check and for unlisted folders).
_AUDIO_TO_LABEL = {1: config.LABEL_INFESTED, 0: config.LABEL_CLEAN}

#: Canonical CSV column names (case-insensitive match against the real header).
_COL_FOLDER = "FOLDER"
_COL_AUDIO = "AUDIO"


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
    """Download + unzip a Kaggle dataset via the official API."""
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
    api.authenticate()
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

    Tries local → Kaggle → URL, each falling back to its ``config`` value so
    ``build_combined`` (which passes no args) honours the env-driven settings.
    """
    local = local or config.TREEVIBES_LOCAL
    if local:
        src = Path(local).expanduser()
        if src.is_dir():
            return src
        if src.is_file():
            return _extract(src, work_dir / "extracted")
        raise RuntimeError(f"TREEVIBES_LOCAL={src} does not exist.")

    kaggle = kaggle or config.TREEVIBES_KAGGLE
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
# Annotation CSV parsing.
# --------------------------------------------------------------------------------------

def _find_annotation_csv(root: Path) -> Path | None:
    """Locate the annotation CSV under ``root`` (first plausible match)."""
    candidates = sorted(root.rglob("*.csv"))
    for csv_path in candidates:
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as fh:
                header = next(csv.reader(fh), [])
        except (OSError, StopIteration):
            continue
        upper = {h.strip().upper() for h in header}
        if _COL_FOLDER in upper and _COL_AUDIO in upper:
            return csv_path
    return candidates[0] if candidates else None


def _folder_number(name: str) -> int | None:
    """Extract the trailing folder number, e.g. 'folder_12' -> 12, '7' -> 7."""
    m = re.search(r"(\d+)\s*$", str(name))
    return int(m.group(1)) if m else None


@dataclass
class FolderInfo:
    """Per-folder annotation joined to the published lists."""

    folder: int
    audio: int | None          # raw AUDIO column value (None if missing/unparsable)
    label: str | None          # resolved label, or None if undeterminable
    list_membership: str | None  # 'infested' | 'clean' | None (unlisted)
    audio_disagrees: bool      # listed folder whose AUDIO contradicts the list


def parse_annotations(csv_path: Path) -> dict[int, FolderInfo]:
    """Parse the annotation CSV into ``{folder_number: FolderInfo}``.

    Resolution per spec:
      * Folder in a published list → label from the list; cross-check AUDIO and
        flag (count) any disagreement, but the list wins.
      * Folder not in either list → label from AUDIO (1→infested, 0→clean).
      * Unrecognised/missing AUDIO for an unlisted folder → label None (reported).
    """
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        # Case-insensitive column lookup.
        colmap = {(c or "").strip().upper(): c for c in (reader.fieldnames or [])}
        folder_col = colmap.get(_COL_FOLDER)
        audio_col = colmap.get(_COL_AUDIO)
        if folder_col is None:
            raise ValueError(
                f"Annotation CSV {csv_path} has no FOLDER column "
                f"(found: {reader.fieldnames})."
            )

        infos: dict[int, FolderInfo] = {}
        for row in reader:
            folder = _folder_number(row.get(folder_col, ""))
            if folder is None:
                continue
            audio = _parse_int(row.get(audio_col)) if audio_col else None

            if folder in INFESTED_FOLDERS:
                membership = config.LABEL_INFESTED
            elif folder in CLEAN_FOLDERS:
                membership = config.LABEL_CLEAN
            else:
                membership = None

            if membership is not None:
                label = membership
                disagrees = audio in _AUDIO_TO_LABEL and _AUDIO_TO_LABEL[audio] != membership
            else:
                label = _AUDIO_TO_LABEL.get(audio) if audio is not None else None
                disagrees = False

            infos[folder] = FolderInfo(
                folder=folder,
                audio=audio,
                label=label,
                list_membership=membership,
                audio_disagrees=disagrees,
            )
    return infos


def _parse_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------------------
# Indexing extracted audio + annotations -> ManifestRows + a drop report.
# --------------------------------------------------------------------------------------

@dataclass
class IngestReport:
    """Accounting for every wav seen, so nothing is dropped silently."""

    total_wavs: int = 0
    kept: int = 0
    dropped_no_csv_row: int = 0
    dropped_unparsable_folder: int = 0
    dropped_unrecognized_audio: int = 0
    audio_list_disagreements: int = 0
    unlisted_folders_labelled_by_audio: int = 0
    by_label: dict[str, int] = field(default_factory=dict)
    sites: set[str] = field(default_factory=set)
    dropped_examples: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "total_wavs": self.total_wavs,
            "kept": self.kept,
            "by_label": dict(self.by_label),
            "unique_sites": len(self.sites),
            "dropped_no_csv_row": self.dropped_no_csv_row,
            "dropped_unparsable_folder": self.dropped_unparsable_folder,
            "dropped_unrecognized_audio": self.dropped_unrecognized_audio,
            "audio_list_disagreements": self.audio_list_disagreements,
            "unlisted_folders_labelled_by_audio": self.unlisted_folders_labelled_by_audio,
            "dropped_examples": self.dropped_examples[:10],
        }


def index(extracted: Path, annotations: dict[int, FolderInfo]) -> tuple[list[ManifestRow], IngestReport]:
    """Join wavs to folder annotations into ManifestRows + an accounting report.

    Pure filesystem + dict logic (no network), so it is unit-testable.
    """
    report = IngestReport()
    rows: list[ManifestRow] = []
    disagreeing_folders: set[int] = set()
    audio_labelled_folders: set[int] = set()

    for audio_path in sorted(extracted.rglob("*")):
        if not audio_path.is_file() or audio_path.suffix.lower() not in AUDIO_EXTS:
            continue
        report.total_wavs += 1

        folder = _folder_number(audio_path.parent.name)
        if folder is None:
            report.dropped_unparsable_folder += 1
            _note_drop(report, audio_path)
            continue

        info = annotations.get(folder)
        if info is None:
            report.dropped_no_csv_row += 1
            _note_drop(report, audio_path)
            continue
        if info.label is None:
            report.dropped_unrecognized_audio += 1
            _note_drop(report, audio_path)
            continue

        if info.audio_disagrees:
            disagreeing_folders.add(folder)
        if info.list_membership is None:
            audio_labelled_folders.add(folder)

        signal, sr = audio_io.read_wav(audio_path)
        site = str(folder)  # FOLDER number == site (one folder ~ one tree)
        rows.append(
            ManifestRow(
                path=str(audio_path),
                label=info.label,
                source="treevibes",
                site=site,
                sample_rate=config.SAMPLE_RATE,
                duration=round(len(signal) / sr, 4),
            )
        )
        report.kept += 1
        report.by_label[info.label] = report.by_label.get(info.label, 0) + 1
        report.sites.add(site)

    # Count folders (not clips) for the structural flags.
    report.audio_list_disagreements = len(disagreeing_folders)
    report.unlisted_folders_labelled_by_audio = len(audio_labelled_folders)
    return rows, report


def _note_drop(report: IngestReport, path: Path) -> None:
    if len(report.dropped_examples) < 10:
        report.dropped_examples.append(str(path))


# --------------------------------------------------------------------------------------
# Public API.
# --------------------------------------------------------------------------------------

def build_rows_with_report(
    url: str | None = None,
    kaggle: str | None = None,
    local: str | None = None,
    work_dir: Path | None = None,
    csv_path: Path | None = None,
) -> tuple[list[ManifestRow], IngestReport]:
    """Acquire + index TreeVibes into ManifestRows, returning the drop report too.

    Raises:
        RuntimeError: no source configured, no annotation CSV, or zero labelled clips.
    """
    work_dir = work_dir or (config.PATHS.data_dir / "treevibes")
    extracted = _resolve_source(url, kaggle, local, work_dir)

    csv_path = csv_path or _find_annotation_csv(extracted)
    if csv_path is None:
        raise RuntimeError(
            f"No annotation CSV found under {extracted}. TreeVibes ingest is "
            "CSV-driven (columns FOLDER, IMEI, ..., AUDIO, ...); include it."
        )
    annotations = parse_annotations(csv_path)
    rows, report = index(extracted, annotations)
    if not rows:
        raise RuntimeError(
            f"No labelled audio produced from {extracted} using {csv_path.name}. "
            f"Report: {report.summary()}"
        )
    return rows, report


def build_rows(
    url: str | None = None,
    kaggle: str | None = None,
    local: str | None = None,
    work_dir: Path | None = None,
    csv_path: Path | None = None,
) -> list[ManifestRow]:
    """Acquire + index TreeVibes into ManifestRows (drop report discarded)."""
    rows, _ = build_rows_with_report(url, kaggle, local, work_dir, csv_path)
    return rows


def build_manifest(
    url: str | None = None,
    kaggle: str | None = None,
    local: str | None = None,
    work_dir: Path | None = None,
    csv_path: Path | None = None,
):
    """Build a manifest from TreeVibes alone."""
    rows, _ = build_rows_with_report(url, kaggle, local, work_dir, csv_path)
    return write_manifest(rows)
