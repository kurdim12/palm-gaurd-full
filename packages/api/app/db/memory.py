"""In-memory backend — default when Supabase is not configured.

Thread-unsafe by design (single-process dev/test). Seeds a few demo trees so the
dashboard and map have something to render out of the box.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..models import Alert, Detection, StatusCounts, Tree, TreeStatus


def _demo_trees() -> list[Tree]:
    """A small grove in Al-Ahsa, Saudi Arabia (a real date-palm region)."""
    base_lat, base_lon = 25.383, 49.586
    specs = [
        ("نخلة 1", "Palm 1", 0.0000, 0.0000, TreeStatus.CLEAN),
        ("نخلة 2", "Palm 2", 0.0008, 0.0005, TreeStatus.CLEAN),
        ("نخلة 3", "Palm 3", 0.0003, 0.0012, TreeStatus.SUSPECT),
        ("نخلة 4", "Palm 4", 0.0015, 0.0009, TreeStatus.INFESTED),
        ("نخلة 5", "Palm 5", 0.0011, 0.0018, TreeStatus.CLEAN),
    ]
    trees = []
    for i, (ar, en, dlat, dlon, status) in enumerate(specs, start=1):
        trees.append(
            Tree(
                id=f"tree-{i:03d}",
                name_ar=ar,
                name_en=en,
                lat=base_lat + dlat,
                lon=base_lon + dlon,
                status=status,
            )
        )
    return trees


class InMemoryDatabase:
    """Dict-backed implementation of :class:`Database`."""

    def __init__(self, seed_demo: bool = True) -> None:
        self._trees: dict[str, Tree] = {}
        self._detections: list[Detection] = []
        self._alerts: list[Alert] = []
        if seed_demo:
            for tree in _demo_trees():
                self._trees[tree.id] = tree

    # --- trees ---
    def get_tree(self, tree_id: str) -> Tree | None:
        return self._trees.get(tree_id)

    def upsert_tree(self, tree: Tree) -> Tree:
        self._trees[tree.id] = tree
        return tree

    def list_trees(self) -> list[Tree]:
        return list(self._trees.values())

    def status_counts(self) -> StatusCounts:
        counts = StatusCounts()
        for tree in self._trees.values():
            setattr(counts, tree.status.value, getattr(counts, tree.status.value) + 1)
        return counts

    # --- detections ---
    def add_detection(self, detection: Detection) -> Detection:
        self._detections.append(detection)
        return detection

    def list_detections(self, tree_id: str, limit: int = 100) -> list[Detection]:
        items = [d for d in self._detections if d.tree_id == tree_id]
        items.sort(key=lambda d: d.captured_at, reverse=True)
        return items[:limit]

    # --- alerts ---
    def add_alert(self, alert: Alert) -> Alert:
        self._alerts.append(alert)
        return alert

    def list_alerts(self, only_open: bool = False, limit: int = 100) -> list[Alert]:
        items = [a for a in self._alerts if not (only_open and a.acknowledged)]
        items.sort(key=lambda a: a.created_at, reverse=True)
        return items[:limit]

    def acknowledge_alert(self, alert_id: str) -> Alert | None:
        for i, alert in enumerate(self._alerts):
            if alert.id == alert_id:
                updated = alert.model_copy(
                    update={"acknowledged": True,
                            "acknowledged_at": datetime.now(timezone.utc)}
                )
                self._alerts[i] = updated
                return updated
        return None

    def latest_alert_for_tree(self, tree_id: str) -> Alert | None:
        candidates = [a for a in self._alerts if a.tree_id == tree_id]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.created_at)
