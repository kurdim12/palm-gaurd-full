"""Alerts inbox endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_db
from ..models import Alert

router = APIRouter(prefix="/api/v1", tags=["alerts"])


@router.get("/alerts", response_model=list[Alert])
def list_alerts(only_open: bool = False, limit: int = 100, db=Depends(get_db)) -> list[Alert]:
    """Alerts newest-first; ``only_open=true`` hides acknowledged ones."""
    return db.list_alerts(only_open=only_open, limit=limit)


@router.post("/alerts/{alert_id}/acknowledge", response_model=Alert)
def acknowledge_alert(alert_id: str, db=Depends(get_db)) -> Alert:
    """Acknowledge (close) an alert."""
    alert = db.acknowledge_alert(alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Unknown alert")
    return alert
