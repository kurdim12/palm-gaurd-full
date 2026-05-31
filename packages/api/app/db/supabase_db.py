"""Supabase (PostgREST) backend.

Mirrors :class:`InMemoryDatabase` against the schema in ``schema.sql``. The
``supabase`` client is imported lazily so the in-memory path needs no extra deps.
Apply ``schema.sql`` to your project before using this backend (see README).
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import Alert, Detection, StatusCounts, Tree, TreeStatus


class SupabaseDatabase:
    """PostgREST-backed implementation of :class:`Database`."""

    def __init__(self, url: str, service_key: str) -> None:
        from supabase import create_client  # noqa: PLC0415 — optional dep

        self._client = create_client(url, service_key)

    # --- trees ---
    def get_tree(self, tree_id: str) -> Tree | None:
        res = self._client.table("trees").select("*").eq("id", tree_id).limit(1).execute()
        rows = res.data or []
        return Tree(**rows[0]) if rows else None

    def upsert_tree(self, tree: Tree) -> Tree:
        self._client.table("trees").upsert(_jsonable(tree)).execute()
        return tree

    def list_trees(self) -> list[Tree]:
        res = self._client.table("trees").select("*").execute()
        return [Tree(**row) for row in (res.data or [])]

    def status_counts(self) -> StatusCounts:
        counts = StatusCounts()
        for tree in self.list_trees():
            setattr(counts, tree.status.value, getattr(counts, tree.status.value) + 1)
        return counts

    # --- detections ---
    def add_detection(self, detection: Detection) -> Detection:
        self._client.table("detections").insert(_jsonable(detection)).execute()
        return detection

    def list_detections(self, tree_id: str, limit: int = 100) -> list[Detection]:
        res = (
            self._client.table("detections")
            .select("*")
            .eq("tree_id", tree_id)
            .order("captured_at", desc=True)
            .limit(limit)
            .execute()
        )
        return [Detection(**row) for row in (res.data or [])]

    # --- alerts ---
    def add_alert(self, alert: Alert) -> Alert:
        self._client.table("alerts").insert(_jsonable(alert)).execute()
        return alert

    def list_alerts(self, only_open: bool = False, limit: int = 100) -> list[Alert]:
        query = self._client.table("alerts").select("*")
        if only_open:
            query = query.eq("acknowledged", False)
        res = query.order("created_at", desc=True).limit(limit).execute()
        return [Alert(**row) for row in (res.data or [])]

    def acknowledge_alert(self, alert_id: str) -> Alert | None:
        res = (
            self._client.table("alerts")
            .update({"acknowledged": True,
                     "acknowledged_at": datetime.now(timezone.utc).isoformat()})
            .eq("id", alert_id)
            .execute()
        )
        rows = res.data or []
        return Alert(**rows[0]) if rows else None

    def latest_alert_for_tree(self, tree_id: str) -> Alert | None:
        res = (
            self._client.table("alerts")
            .select("*")
            .eq("tree_id", tree_id)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        return Alert(**rows[0]) if rows else None


def _jsonable(model) -> dict:
    """Serialise a pydantic model to JSON-safe primitives for PostgREST."""
    return model.model_dump(mode="json")
