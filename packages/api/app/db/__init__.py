"""Database layer: a small interface with in-memory and Supabase backends."""

from __future__ import annotations

from ..config import get_settings
from .base import Database
from .memory import InMemoryDatabase


def get_database() -> Database:
    """Return the configured backend: Supabase if env is set, else in-memory."""
    settings = get_settings()
    if settings.use_supabase:
        from .supabase_db import SupabaseDatabase

        return SupabaseDatabase(settings.supabase_url, settings.supabase_service_key)
    return InMemoryDatabase()


__all__ = ["Database", "InMemoryDatabase", "get_database"]
