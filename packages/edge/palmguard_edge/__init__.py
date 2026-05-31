"""Palm Guard edge agent: capture -> on-device inference -> store-and-forward.

Adds the sibling ``packages/ml`` to ``sys.path`` on import so the edge uses the
*exact* training feature/inference code (CLAUDE.md: mirror training in
inference), never a re-implementation. Must run before any ``palmguard_ml``
import in this package.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ML_PKG = Path(__file__).resolve().parents[1].parent / "ml"
if _ML_PKG.exists() and str(_ML_PKG) not in sys.path:
    sys.path.insert(0, str(_ML_PKG))
