"""End-to-end synthetic pipeline: data → manifest → baseline → inference.

Exercises the real ingest + dataset + baseline path on a tiny synthetic set so it
stays fast. Writes into a temp data dir via monkeypatched paths.
"""

from __future__ import annotations

import numpy as np

from palmguard_ml import config, features
from palmguard_ml.ingest import synthetic
from palmguard_ml.manifest import read_manifest, summary, validate


def test_synthetic_build_manifest_is_valid(tmp_path, monkeypatch):
    # Point all derived data dirs under tmp_path. Modules read config.PATHS at
    # call time, so patching the singleton is enough.
    monkeypatch.setattr(config, "PATHS", config.Paths(root=tmp_path))

    manifest = synthetic.build_manifest(n_sites_per_class=3, clips_per_site=4, seed=1)
    rows = read_manifest(manifest)
    validate(rows)

    s = summary(rows)
    assert s["clips"] == 2 * 3 * 4
    assert s["by_label"]["clean"] > 0 and s["by_label"]["infested"] > 0
    assert all(not np.isnan(config.LABEL_TO_INT[r.label]) for r in rows)


def test_baseline_separates_synthetic_classes(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PATHS", config.Paths(root=tmp_path))
    synthetic.build_manifest(n_sites_per_class=6, clips_per_site=8, seed=2)

    # Import after patching so dataset/baseline see the patched PATHS.
    from palmguard_ml.baseline import train_and_eval

    _, metrics = train_and_eval()
    # Synthetic classes are designed to be separable; baseline must clear the
    # spec's recall-first bar on the held-out SITE split.
    assert metrics.infested_recall >= 0.9
    assert metrics.pr_auc >= 0.85


def test_feature_path_matches_between_calls(infested_clip):
    # Determinism: same input -> identical features (mirror-training guarantee).
    a = features.feature_vectors_from_audio(infested_clip)
    b = features.feature_vectors_from_audio(infested_clip)
    assert np.array_equal(a, b)
