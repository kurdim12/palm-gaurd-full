"""Detection ingest endpoint — the edge → cloud entrypoint.

POST a classification result; the status engine debounces it, the tree state is
updated, and an alert is raised (and possibly sent) on a confirmed new
infestation.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_db, get_dispatcher, get_engine
from ..models import (
    Detection,
    DetectionIn,
    DetectionResult,
    Tree,
)

router = APIRouter(prefix="/api/v1", tags=["detections"])


@router.post("/detections", response_model=DetectionResult, status_code=201)
def create_detection(
    payload: DetectionIn,
    db=Depends(get_db),
    engine=Depends(get_engine),
    dispatcher=Depends(get_dispatcher),
) -> DetectionResult:
    """Ingest one edge detection and update tree status."""
    tree = db.get_tree(payload.tree_id)
    if tree is None:
        # Unknown tree: auto-register at origin so detections are never dropped.
        tree = Tree(id=payload.tree_id, lat=0.0, lon=0.0)
        db.upsert_tree(tree)

    detection = Detection(**payload.model_dump())
    db.add_detection(detection)

    decision = engine.apply(tree, detection)
    db.upsert_tree(decision.tree)

    alert = dispatcher.maybe_alert(decision.tree) if decision.became_infested else None

    return DetectionResult(
        detection=detection,
        tree=decision.tree,
        status_changed=decision.changed,
        alert=alert,
    )


@router.get("/trees/{tree_id}/detections", response_model=list[Detection])
def list_detections(tree_id: str, limit: int = 100, db=Depends(get_db)) -> list[Detection]:
    """Recent detections for a tree, newest first."""
    if db.get_tree(tree_id) is None:
        raise HTTPException(status_code=404, detail="Unknown tree")
    return db.list_detections(tree_id, limit=limit)
