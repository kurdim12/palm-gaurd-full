"""Manifest validation + the site-level split contract (golden rule #3)."""

from __future__ import annotations

import pytest

from palmguard_ml import config
from palmguard_ml.manifest import (
    ManifestRow,
    site_split,
    summary,
    validate,
)


def _row(label: str, site: str) -> ManifestRow:
    return ManifestRow(
        path=f"data/raw/{label}/{site}/x.wav",
        label=label,
        source="synthetic",
        site=site,
        sample_rate=config.SAMPLE_RATE,
        duration=config.CLIP_DURATION_S,
    )


def _balanced_rows(n_sites: int = 6, clips: int = 4) -> list[ManifestRow]:
    rows = []
    for label in config.CLASSES:
        for s in range(n_sites):
            for _ in range(clips):
                rows.append(_row(label, f"{label}-site-{s}"))
    return rows


def test_validate_rejects_single_class():
    rows = [_row("clean", "c0"), _row("clean", "c1")]
    with pytest.raises(ValueError):
        validate(rows)


def test_validate_rejects_unknown_label():
    rows = [_row("clean", "c0"), _row("mystery", "m0")]
    with pytest.raises(ValueError):
        validate(rows)


def test_site_split_does_not_leak_sites():
    rows = _balanced_rows()
    train, test = site_split(rows)
    train_sites = {r.site for r in train}
    test_sites = {r.site for r in test}
    assert train_sites and test_sites
    assert train_sites.isdisjoint(test_sites)  # no tree spans both splits


def test_site_split_keeps_both_classes_each_side():
    rows = _balanced_rows()
    train, test = site_split(rows)
    assert {r.label for r in train} == set(config.CLASSES)
    assert {r.label for r in test} == set(config.CLASSES)


def test_site_split_is_deterministic():
    rows = _balanced_rows()
    a_train, a_test = site_split(rows)
    b_train, b_test = site_split(rows)
    assert [r.site for r in a_test] == [r.site for r in b_test]
    assert [r.site for r in a_train] == [r.site for r in b_train]


def test_summary_counts():
    rows = _balanced_rows(n_sites=3, clips=2)
    s = summary(rows)
    assert s["clips"] == 3 * 2 * 2
    assert s["by_label"]["clean"] == 6
    assert s["by_label"]["infested"] == 6
