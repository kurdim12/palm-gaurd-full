"""Shared dependency singletons (DB, settings, engine, dispatcher).

A single process-wide DB instance keeps the in-memory backend coherent across
requests. Tests override :func:`get_db` to inject a fresh DB.
"""

from __future__ import annotations

from functools import lru_cache

from .alerts import AlertDispatcher
from .config import Settings, get_settings
from .db import Database, get_database
from .status_engine import StatusEngine


@lru_cache
def get_db() -> Database:
    return get_database()


@lru_cache
def get_engine() -> StatusEngine:
    s = get_settings()
    return StatusEngine(
        streak=s.status_infested_streak,
        confidence_threshold=s.status_confidence_threshold,
    )


def get_dispatcher() -> AlertDispatcher:
    return AlertDispatcher(get_settings(), get_db())


def get_settings_dep() -> Settings:
    return get_settings()
