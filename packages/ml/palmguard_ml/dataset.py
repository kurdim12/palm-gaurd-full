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
    """Materialised per-window features for one split.

    Each row corresponds to one 1 s window; ``files`` records which manifest file
    each window came from, so window scores can be aggregated to a file/tree
    decision at evaluation time.
    """

    X_vec: np.ndarray       # (n_windows, n_features) classical features
    X_cnn: np.ndarray       # (n_windows, n_mels, n_time, 1) log-mel images
    y: np.ndarray           # (n_windows,) int labels (inherited from the file)
    sites: list[str]        # per-window site id
    files: list[str]        # per-window source-file id (manifest path)
    rows: list[ManifestRow]

    def __len__(self) -> int:
        return len(self.y)


def _load_file(row: ManifestRow) -> np.ndarray:
    path = config.PATHS.root / row.path
    if not path.exists():
        path = type(path)(row.path)  # absolute fallback
    signal, sr = audio_io.read_wav(path)
    from . import dsp

    return dsp.preprocess(signal, sr)


def _materialise(rows: list[ManifestRow], want_cnn: bool) -> Dataset:
    from . import dsp

    vecs: list[np.ndarray] = []
    cnns: list[np.ndarray] = []
    ys: list[int] = []
    sites: list[str] = []
    files: list[str] = []
    for row in rows:
        pre = _load_file(row)
        label = config.LABEL_TO_INT[row.label]
        for win in dsp.windows(pre):
            vecs.append(features.feature_vector(win))
            if want_cnn:
                cnns.append(features.cnn_input(win))
            ys.append(label)
            sites.append(row.site)
            files.append(row.path)
    X_vec = np.stack(vecs) if vecs else np.empty((0, len(features.FEATURE_NAMES)), np.float32)
    X_cnn = (
        np.stack(cnns)
        if (want_cnn and cnns)
        else np.empty((0, config.N_MELS, config.N_TIME_FRAMES, 1), np.float32)
    )
    return Dataset(X_vec, X_cnn, np.asarray(ys, dtype=np.int64), sites, files, rows)


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
