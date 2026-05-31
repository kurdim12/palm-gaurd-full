"""Tree + status endpoints powering the map and tree-detail views."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_db
from ..models import StatusCounts, Tree, TreeStatus

router = APIRouter(prefix="/api/v1", tags=["trees"])


@router.get("/trees", response_model=list[Tree])
def list_trees(db=Depends(get_db)) -> list[Tree]:
    """All monitored trees (for the farm map)."""
    return db.list_trees()


@router.get("/trees/{tree_id}", response_model=Tree)
def get_tree(tree_id: str, db=Depends(get_db)) -> Tree:
    tree = db.get_tree(tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Unknown tree")
    return tree


@router.get("/status/counts", response_model=StatusCounts)
def status_counts(db=Depends(get_db)) -> StatusCounts:
    """Aggregate tree counts by status (dashboard summary cards)."""
    return db.status_counts()


@router.post("/trees/{tree_id}/treat", response_model=Tree)
def mark_treated(tree_id: str, db=Depends(get_db)) -> Tree:
    """Mark a tree as treated and reset its infested streak."""
    tree = db.get_tree(tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail="Unknown tree")
    updated = tree.model_copy(
        update={
            "status": TreeStatus.TREATED,
            "infested_streak": 0,
            "updated_at": datetime.now(timezone.utc),
        }
    )
    return db.upsert_tree(updated)
