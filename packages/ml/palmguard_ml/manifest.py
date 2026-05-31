"""Manifest schema, validation, and the **site-level** train/test split.

The manifest is the contract between ingest and everything downstream. Whatever
the dataset (synthetic, TreeVibes, future sources), it must yield a CSV with
:data:`config.MANIFEST_COLUMNS`. Downstream code never looks at raw dataset
layout — only the manifest.

CLAUDE.md golden rule #3: **split by site, never by clip.**
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import config


@dataclass(frozen=True)
class ManifestRow:
    """One audio clip's metadata."""

    path: str
    label: str
    source: str
    site: str
    sample_rate: int
    duration: float

    def as_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "label": self.label,
            "source": self.source,
            "site": self.site,
            "sample_rate": str(self.sample_rate),
            "duration": f"{self.duration:.4f}",
        }


def write_manifest(rows: list[ManifestRow], path: Path | None = None) -> Path:
    """Write rows to the manifest CSV (default: :data:`config.PATHS.manifest`)."""
    path = path or config.PATHS.manifest
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(config.MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())
    return path


def read_manifest(path: Path | None = None) -> list[ManifestRow]:
    """Read and validate a manifest CSV."""
    path = path or config.PATHS.manifest
    if not path.exists():
        raise FileNotFoundError(
            f"No manifest at {path}. Run `make data` (or wire TREEVIBES_URL) first."
        )
    rows: list[ManifestRow] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or tuple(reader.fieldnames) != config.MANIFEST_COLUMNS:
            raise ValueError(
                f"Manifest schema mismatch. Expected {config.MANIFEST_COLUMNS}, "
                f"got {reader.fieldnames}."
            )
        for raw in reader:
            rows.append(
                ManifestRow(
                    path=raw["path"],
                    label=raw["label"],
                    source=raw["source"],
                    site=raw["site"],
                    sample_rate=int(raw["sample_rate"]),
                    duration=float(raw["duration"]),
                )
            )
    return rows


def validate(rows: list[ManifestRow]) -> None:
    """Raise if the manifest is unusable (empty, NaN labels, single class)."""
    if not rows:
        raise ValueError("Manifest is empty.")
    labels = set()
    for r in rows:
        if r.label not in config.CLASSES:
            raise ValueError(f"Unknown/NaN label {r.label!r} for {r.path}.")
        labels.add(r.label)
    missing = set(config.CLASSES) - labels
    if missing:
        raise ValueError(f"Manifest missing class(es): {sorted(missing)}.")


def _site_bucket(site: str) -> float:
    """Deterministically map a site id to [0, 1) via a stable hash."""
    digest = hashlib.sha256(site.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _dominant_label(rows: list[ManifestRow], site: str) -> str:
    """The majority label among a site's clips (sites are usually single-class)."""
    counts: dict[str, int] = {}
    for r in rows:
        if r.site == site:
            counts[r.label] = counts.get(r.label, 0) + 1
    return max(counts, key=counts.get)


def site_split(
    rows: list[ManifestRow],
    test_fraction: float = config.TEST_SITE_FRACTION,
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Split rows into (train, test) so no *site* spans both.

    Site selection is **stratified by class**: within each class we hold out a
    deterministic fraction of *sites* (chosen by a stable hash of the site id).
    This keeps clips from one tree on a single side (golden rule #3) while
    guaranteeing both classes appear in both splits. Deterministic across runs.
    """
    sites = sorted({r.site for r in rows})
    by_class: dict[str, list[str]] = {}
    for site in sites:
        by_class.setdefault(_dominant_label(rows, site), []).append(site)

    test_sites: set[str] = set()
    for label, label_sites in by_class.items():
        # Rank this class's sites by hash, take the lowest `test_fraction` as test,
        # but always keep >=1 site on each side when the class has >=2 sites.
        ranked = sorted(label_sites, key=_site_bucket)
        n_test = int(round(len(ranked) * test_fraction))
        if len(ranked) >= 2:
            n_test = max(1, min(n_test, len(ranked) - 1))
        test_sites.update(ranked[:n_test])

    train = [r for r in rows if r.site not in test_sites]
    test = [r for r in rows if r.site in test_sites]

    for name, split in (("train", train), ("test", test)):
        present = {r.label for r in split}
        if split and len(present) < 2:
            raise ValueError(
                f"{name} split has only class(es) {present}. Need more sites per class "
                "for a valid site-level split."
            )
    return train, test


def summary(rows: list[ManifestRow]) -> dict[str, object]:
    """Human-readable counts for logging."""
    by_label: dict[str, int] = {c: 0 for c in config.CLASSES}
    sites_by_label: dict[str, set[str]] = {c: set() for c in config.CLASSES}
    for r in rows:
        by_label[r.label] += 1
        sites_by_label[r.label].add(r.site)
    return {
        "clips": len(rows),
        "sites": len({r.site for r in rows}),
        "by_label": by_label,
        "sites_by_label": {k: len(v) for k, v in sites_by_label.items()},
        "sources": sorted({r.source for r in rows}),
    }
