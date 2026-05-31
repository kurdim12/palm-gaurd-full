"""Central configuration for Palm Guard ML.

**Scientific constants in this module are fixed** (see CLAUDE.md, golden rule #1).
They derive from the Red Palm Weevil (RPW) bio-acoustics literature: larval feeding
and locomotion inside the date-palm trunk produce brief, broadband *bursts* whose
energy concentrates in a band roughly 200 Hz–2.5 kHz with a perceptual peak near
2.25 kHz. Do not "tune" these to make metrics look better, and never inline these
numbers elsewhere — import them from here.

Everything downstream of the manifest is dataset-agnostic; only paths and the
optional TreeVibes URL are environment-dependent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------
# Fixed acoustic constants — RPW literature. DO NOT CHANGE to chase metrics.
# --------------------------------------------------------------------------------------

#: Working sample rate (Hz). Contact-sensor audio is resampled to this everywhere.
SAMPLE_RATE: int = 8_000

#: Passband for the RPW signal (Hz). Energy outside this band is environmental noise.
BAND_LOW_HZ: int = 200
BAND_HIGH_HZ: int = 2_500

#: Perceptual / spectral peak of larval bursts (Hz).
PEAK_HZ: int = 2_250

#: Larval feeding burst structure. Bursts are short transients separated by gaps.
BURST_MIN_MS: float = 3.0   # shortest credible feeding transient
BURST_MAX_MS: float = 40.0  # longest credible single burst
#: Typical inter-burst interval range (s) for an active larva.
BURST_INTERVAL_MIN_S: float = 0.05
BURST_INTERVAL_MAX_S: float = 1.5

# --------------------------------------------------------------------------------------
# Fixed framing / feature geometry (derived from the constants above).
# --------------------------------------------------------------------------------------

#: Canonical analysis clip length (s). Clips are cropped/padded to this.
CLIP_DURATION_S: float = 2.0

#: STFT framing.
FRAME_LENGTH: int = 256          # 32 ms @ 8 kHz
HOP_LENGTH: int = 128            # 16 ms @ 8 kHz, 50% overlap
N_FFT: int = 256

#: Log-mel spectrogram geometry fed to the CNN.
N_MELS: int = 64
MEL_FMIN_HZ: int = BAND_LOW_HZ
MEL_FMAX_HZ: int = BAND_HIGH_HZ

#: Fixed number of time frames the CNN expects (clip cropped/padded to this).
N_TIME_FRAMES: int = 1 + int((SAMPLE_RATE * CLIP_DURATION_S - FRAME_LENGTH) // HOP_LENGTH)

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

#: Set to enable the real TreeVibes ingest path. Empty string => synthetic only.
TREEVIBES_URL: str = os.environ.get("TREEVIBES_URL", "")

#: CNN backbone selector consumed by model.py. Kept here so experiments are explicit.
CNN_BACKBONE: str = os.environ.get("PALMGUARD_BACKBONE", "small_cnn")

RANDOM_SEED: int = 42

#: Fraction of *sites* (never clips) held out for test.
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
