"""Pydantic request/response schemas + domain enums."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid4())


class TreeStatus(str, Enum):
    """Lifecycle of a monitored tree."""

    CLEAN = "clean"
    SUSPECT = "suspect"      # noisy / inconclusive infested hits, not yet confirmed
    INFESTED = "infested"
    TREATED = "treated"


class Label(str, Enum):
    CLEAN = "clean"
    INFESTED = "infested"


class DetectionIn(BaseModel):
    """A single edge classification result uploaded to the API."""

    device_id: str
    tree_id: str
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    captured_at: datetime = Field(default_factory=_now)
    audio_url: str | None = None
    model_version: str | None = None


class Detection(DetectionIn):
    id: str = Field(default_factory=_uuid)
    received_at: datetime = Field(default_factory=_now)


class Tree(BaseModel):
    id: str
    name_ar: str = ""
    name_en: str = ""
    lat: float
    lon: float
    status: TreeStatus = TreeStatus.CLEAN
    infested_streak: int = 0
    last_detection_at: datetime | None = None
    updated_at: datetime = Field(default_factory=_now)


class Alert(BaseModel):
    id: str = Field(default_factory=_uuid)
    tree_id: str
    created_at: datetime = Field(default_factory=_now)
    acknowledged: bool = False
    acknowledged_at: datetime | None = None
    message_ar: str = ""
    message_en: str = ""


class StatusCounts(BaseModel):
    clean: int = 0
    suspect: int = 0
    infested: int = 0
    treated: int = 0


class DetectionResult(BaseModel):
    """Response to a detection upload: the stored detection + resulting tree state."""

    detection: Detection
    tree: Tree
    status_changed: bool
    alert: Alert | None = None
