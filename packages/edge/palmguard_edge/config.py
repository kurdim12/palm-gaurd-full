"""Edge device configuration (env-driven).

The ``packages/ml`` import path is wired in :mod:`palmguard_edge` (the package
``__init__``) so the edge reuses the exact training feature/inference code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_EDGE_ROOT = Path(__file__).resolve().parents[1]
_ML_PKG = _EDGE_ROOT.parents[0] / "ml"


@dataclass(frozen=True)
class EdgeConfig:
    """Runtime configuration for the on-device agent."""

    api_url: str = os.environ.get("EDGE_API_URL", "http://localhost:8000")
    device_id: str = os.environ.get("EDGE_DEVICE_ID", "pi-zero-dev")
    tree_id: str = os.environ.get("EDGE_TREE_ID", "tree-001")
    model_path: str = os.environ.get(
        "EDGE_MODEL_PATH", str(_ML_PKG / "artifacts" / "palmguard.tflite")
    )
    queue_path: str = os.environ.get("EDGE_QUEUE_PATH", str(_EDGE_ROOT / "queue.jsonl"))
    capture_interval_s: int = int(os.environ.get("EDGE_INTERVAL_S", "3600"))  # hourly
    threshold: float = float(os.environ.get("EDGE_THRESHOLD", "0.5"))


def load_config() -> EdgeConfig:
    return EdgeConfig()
