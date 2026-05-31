"""Dataset ingest adapters.

Each adapter turns some raw source into valid ManifestRows (see
:mod:`palmguard_ml.manifest`). Add new datasets here; never couple feature or
training code to a specific dataset's layout.

Authoritative sources (BUILD_SPEC §3):
  * ``treevibes`` — primary RPW corpus (infested + clean).
  * ``esc50``     — non-insect hard negatives → clean.
  * ``synthetic`` — always-available offline smoke-test fixture.
"""

from __future__ import annotations

import logging

from .. import config
from ..manifest import ManifestRow, write_manifest

logger = logging.getLogger("palmguard.ingest")


def build_combined():
    """Build one manifest from whatever real sources are configured.

    Degrades gracefully (SPEC §3): each source is attempted only if it is
    configured, and a source that fails to acquire/parse is logged and skipped
    rather than aborting the build. Returns the written manifest path, or ``None``
    if no real source produced rows (caller should fall back to synthetic).
    """
    from . import esc50, treevibes

    treevibes_configured = bool(
        config.TREEVIBES_LOCAL or config.TREEVIBES_KAGGLE or config.TREEVIBES_URL
    )
    sources = [
        ("treevibes", treevibes_configured, treevibes.build_rows),
        ("esc50", bool(config.ESC50_URL), esc50.build_rows),
    ]
    rows: list[ManifestRow] = []
    for name, configured, builder in sources:
        if not configured:
            continue
        try:
            new = builder()
            logger.info("ingest[%s]: %d clips", name, len(new))
            rows.extend(new)
        except Exception as exc:  # noqa: BLE001 - degrade gracefully per SPEC
            logger.warning("ingest[%s] skipped: %s", name, exc)
    if not rows:
        return None
    return write_manifest(rows)
