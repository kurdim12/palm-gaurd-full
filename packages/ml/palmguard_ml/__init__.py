"""Palm Guard ML — acoustic RPW detection pipeline.

Public surface kept intentionally small; import submodules directly for the rest.
Heavy deps (TensorFlow) live behind lazy imports so the core feature/inference
path stays light enough for the edge device.
"""

from __future__ import annotations

from . import config

__all__ = ["config", "__version__"]
__version__ = "0.1.0"
