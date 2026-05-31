"""Turn a manifest into model-ready arrays, honouring the site-split.

This is the dataset-agnostic boundary: it consumes a manifest and emits feature
matrices. It never inspects raw dataset layout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import audio_io, config, features
from .manifest import ManifestRow, read_manifest, site_split, validate


@dataclass
class Dataset:
    """Materialised features for one split."""

    X_vec: np.ndarray       # (n, n_features) classical features
    X_cnn: np.ndarray       # (n, n_mels, n_time, 1) log-mel images
    y: np.ndarray           # (n,) int labels
    sites: list[str]
    rows: list[ManifestRow]

    def __len__(self) -> int:
        return len(self.y)


def _load_clip(row: ManifestRow) -> np.ndarray:
    path = config.PATHS.root / row.path
    if not path.exists():
        path = type(path)(row.path)  # absolute fallback
    signal, sr = audio_io.read_wav(path)
    from . import dsp

    return dsp.preprocess(signal, sr)


def _materialise(rows: list[ManifestRow], want_cnn: bool) -> Dataset:
    vecs: list[np.ndarray] = []
    cnns: list[np.ndarray] = []
    ys: list[int] = []
    sites: list[str] = []
    for row in rows:
        pre = _load_clip(row)
        vecs.append(features.feature_vector(pre))
        if want_cnn:
            cnns.append(features.cnn_input(pre))
        ys.append(config.LABEL_TO_INT[row.label])
        sites.append(row.site)
    X_vec = np.stack(vecs) if vecs else np.empty((0, len(features.FEATURE_NAMES)), np.float32)
    X_cnn = (
        np.stack(cnns)
        if (want_cnn and cnns)
        else np.empty((0, config.N_MELS, config.N_TIME_FRAMES, 1), np.float32)
    )
    return Dataset(X_vec, X_cnn, np.asarray(ys, dtype=np.int64), sites, rows)


def load_splits(
    manifest_rows: list[ManifestRow] | None = None,
    want_cnn: bool = False,
) -> tuple[Dataset, Dataset]:
    """Load (train, test) datasets from the manifest using the site-split.

    Args:
        manifest_rows: Optional pre-read rows (else read from disk).
        want_cnn: Also materialise the log-mel CNN inputs (slower/heavier).

    Returns:
        ``(train, test)`` :class:`Dataset` instances.
    """
    rows = manifest_rows or read_manifest()
    validate(rows)
    train_rows, test_rows = site_split(rows)
    return _materialise(train_rows, want_cnn), _materialise(test_rows, want_cnn)
