"""Store-and-forward uploader: the offline→online no-loss guarantee."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from palmguard_edge.uploader import UploadQueue  # noqa: E402


def _det(i: int) -> dict:
    return {"device_id": "pi", "tree_id": "t1", "label": "infested", "confidence": 0.9, "n": i}


def test_offline_then_online_loses_nothing(tmp_path):
    queue = UploadQueue(tmp_path / "queue.jsonl")
    for i in range(5):
        queue.enqueue(_det(i))

    # Phase 1: API unreachable — every send fails. Nothing should be delivered or lost.
    offline = lambda d: False  # noqa: E731
    delivered = queue.flush(offline)
    assert delivered == 0
    assert len(queue.pending()) == 5  # all still queued

    # Phase 2: API back online — flush delivers everything, in order, exactly once.
    received: list[int] = []

    def online(d: dict) -> bool:
        received.append(d["n"])
        return True

    delivered = queue.flush(online)
    assert delivered == 5
    assert received == [0, 1, 2, 3, 4]
    assert queue.pending() == []


def test_partial_outage_keeps_undelivered(tmp_path):
    queue = UploadQueue(tmp_path / "queue.jsonl")
    for i in range(4):
        queue.enqueue(_det(i))

    # Deliver the first two, then the connection drops.
    calls = {"n": 0}

    def flaky(d: dict) -> bool:
        calls["n"] += 1
        return calls["n"] <= 2

    delivered = queue.flush(flaky)
    assert delivered == 2
    remaining = [d["n"] for d in queue.pending()]
    assert remaining == [2, 3]  # the two undelivered, preserved in order


def test_sender_exception_is_treated_as_offline(tmp_path):
    queue = UploadQueue(tmp_path / "queue.jsonl")
    queue.enqueue(_det(0))

    def boom(d: dict) -> bool:
        raise ConnectionError("network down")

    delivered = queue.flush(boom)
    assert delivered == 0
    assert len(queue.pending()) == 1  # not lost on a raised network error


def test_enqueue_is_durable_across_instances(tmp_path):
    path = tmp_path / "queue.jsonl"
    UploadQueue(path).enqueue(_det(7))
    # A fresh process/instance must see the persisted detection.
    assert UploadQueue(path).pending()[0]["n"] == 7
