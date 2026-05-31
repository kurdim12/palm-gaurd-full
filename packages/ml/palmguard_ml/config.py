"""Central configuration for Palm Guard ML.

**Scientific constants in this module are fixed** (see CLAUDE.md golden rule #1 and
the authoritative `docs/BUILD_SPEC.md`). RPW larvae produce brief impulsive
feeding "snaps"; usable energy sits in a mid band ~200–2500 Hz with an emphasis
near 2250 Hz. Recordings are single-channel at a low rate — **8 kHz is the
canonical working rate** (matches the TreeVibes corpus; do not resample up).
Audio is analysed in **1.0 s windows** (0.5 s hop); per-window scores aggregate to
a clip/tree decision. Never "tune" these to chase metrics, and never inline these
numbers elsewhere — import them from here.

Everything downstream of the manifest is dataset-agnostic; only paths and the
optional dataset URLs are environment-dependent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Fixed acoustic constants — authoritative BUILD_SPEC. DO NOT CHANGE to chase metrics.
# --------------------------------------------------------------------------------------

#: Canonical working sample rate (Hz). Matches TreeVibes; resample sources DOWN to this.
SAMPLE_RATE: int = 8_000

#: Band-pass focus for the RPW signal (Hz). Energy outside is environmental noise.
BAND_LOW_HZ: int = 200
BAND_HIGH_HZ: int = 2_500

#: Known spectral emphasis of larval feeding (Hz); used in features / sanity checks.
PEAK_HZ: int = 2_250

#: Classification window (SPEC: CLIP_SECONDS / HOP_SECONDS). The model classifies
#: one window at a time; window scores aggregate to a file/tree decision.
WINDOW_SEC: float = 1.0
HOP_SEC: float = 0.5
WINDOW_SAMPLES: int = int(SAMPLE_RATE * WINDOW_SEC)   # 8000
HOP_SAMPLES: int = int(SAMPLE_RATE * HOP_SEC)         # 4000

#: Larval feeding burst structure. Bursts are short transients separated by gaps.
BURST_MIN_MS: float = 3.0   # shortest credible feeding transient
BURST_MAX_MS: float = 40.0  # longest credible single burst
#: Typical inter-burst interval range (s) for an active larva.
BURST_INTERVAL_MIN_S: float = 0.05
BURST_INTERVAL_MAX_S: float = 0.25

# --------------------------------------------------------------------------------------
# Fixed log-mel / framing geometry (SPEC: N_FFT / HOP_LENGTH / N_MELS / FMIN / FMAX).
# --------------------------------------------------------------------------------------

#: STFT framing *within* a classification window.
FRAME_LENGTH: int = 1_024
HOP_LENGTH: int = 256
N_FFT: int = 1_024

#: Log-mel spectrogram geometry fed to the CNN. Mel range (100–3000 Hz) is
#: intentionally wider than the band-pass so band edges are represented.
N_MELS: int = 64
MEL_FMIN_HZ: int = 100
MEL_FMAX_HZ: int = 3_000

#: Number of time frames per window the CNN expects.
N_TIME_FRAMES: int = 1 + int((WINDOW_SAMPLES - FRAME_LENGTH) // HOP_LENGTH)

# --------------------------------------------------------------------------------------
# Class labels.
# --------------------------------------------------------------------------------------

LABEL_CLEAN: str = "clean"
LABEL_INFESTED: str = "infested"
CLASSES = (LABEL_CLEAN, LABEL_INFESTED)
#: Positive (minority, costly-to-miss) class.
POSITIVE_LABEL: str = LABEL_INFESTED
LABEL_TO_INT = {LABEL_CLEAN: 0, LABEL_INFESTED: 1}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

#: Default decision threshold. Recall-first: we bias toward catching infested trees.
DEFAULT_THRESHOLD: float = 0.5
#: Minimum acceptable infested recall when selecting/reporting a model (CLAUDE.md #2).
TARGET_INFESTED_RECALL: float = 0.9

# --------------------------------------------------------------------------------------
# Dataset / training (NOT scientific — safe to tune).
# --------------------------------------------------------------------------------------

#: Real dataset sources (BUILD_SPEC §3). Empty => synthetic smoke-test data.
TREEVIBES_URL: str = os.environ.get("TREEVIBES_URL", "")   # primary RPW corpus
ESC50_URL: str = os.environ.get("ESC50_URL", "")           # hard negatives -> clean

#: Length (s) of each generated synthetic clip (yields several 1 s windows).
SYNTH_CLIP_SEC: float = 3.0

#: CNN backbone selector consumed by model.py. Kept here so experiments are explicit.
CNN_BACKBONE: str = os.environ.get("PALMGUARD_BACKBONE", "small_cnn")

RANDOM_SEED: int = 42

#: Fixed fraction of *sites* (never clips/windows) held out for test.
TEST_SITE_FRACTION: float = 0.3


@dataclass(frozen=True)
class Paths:
    """Resolved filesystem layout. All generated artifacts are gitignored."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[3])

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def manifest(self) -> Path:
        return self.raw_dir / "manifest.csv"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "packages" / "ml" / "artifacts"

    @property
    def baseline_model(self) -> Path:
        return self.artifacts_dir / "baseline.joblib"

    @property
    def keras_model(self) -> Path:
        return self.artifacts_dir / "palmguard.keras"

    @property
    def tflite_model(self) -> Path:
        return self.artifacts_dir / "palmguard.tflite"

    @property
    def metrics(self) -> Path:
        return self.artifacts_dir / "metrics.json"

    @property
    def threshold(self) -> Path:
        return self.artifacts_dir / "threshold.json"

    @property
    def parity(self) -> Path:
        return self.artifacts_dir / "parity.json"


PATHS = Paths()

#: Manifest schema — every ingest path must produce exactly these columns.
MANIFEST_COLUMNS = (
    "path",
    "label",
    "source",
    "site",
    "sample_rate",
    "duration",
)
