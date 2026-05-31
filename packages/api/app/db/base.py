"""Database interface shared by the in-memory and Supabase backends.

Keeping this abstract lets the API run with zero external services in dev/test and
swap to Supabase in production with no route changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..models import Alert, Detection, StatusCounts, Tree


class Database(ABC):
    """Persistence contract for trees, detections, and alerts."""

    # --- trees ---
    @abstractmethod
    def get_tree(self, tree_id: str) -> Tree | None: ...

    @abstractmethod
    def upsert_tree(self, tree: Tree) -> Tree: ...

    @abstractmethod
    def list_trees(self) -> list[Tree]: ...

    @abstractmethod
    def status_counts(self) -> StatusCounts: ...

    # --- detections ---
    @abstractmethod
    def add_detection(self, detection: Detection) -> Detection: ...

    @abstractmethod
    def list_detections(self, tree_id: str, limit: int = 100) -> list[Detection]: ...

    # --- alerts ---
    @abstractmethod
    def add_alert(self, alert: Alert) -> Alert: ...

    @abstractmethod
    def list_alerts(self, only_open: bool = False, limit: int = 100) -> list[Alert]: ...

    @abstractmethod
    def acknowledge_alert(self, alert_id: str) -> Alert | None: ...

    @abstractmethod
    def latest_alert_for_tree(self, tree_id: str) -> Alert | None: ...
