"""Store-and-forward detection uploader.

The offline guarantee (CLAUDE.md): the Pi often has no connectivity. Every
detection is *first* appended to a durable on-disk queue (``queue.jsonl``); a
flush attempts to POST queued detections to the API and only removes them after a
confirmed success. A crash or power loss mid-flush therefore never loses a
detection — at worst one is delivered twice (the API treats detections as
append-only, so a duplicate is harmless).

The network transport is injected (``sender``) so the loop is testable without a
real server and trivially swappable (urllib by default, no heavy deps).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

#: A sender takes a detection dict and returns True on confirmed delivery.
Sender = Callable[[dict], bool]


class UploadQueue:
    """Append-only JSONL queue with at-least-once flush semantics."""

    def __init__(self, queue_path: str | Path) -> None:
        self.queue_path = Path(queue_path)
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(self, detection: dict) -> None:
        """Durably append a detection before any network attempt."""
        with self.queue_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(detection, ensure_ascii=False) + "\n")
            fh.flush()

    def pending(self) -> list[dict]:
        """All queued detections (in order)."""
        if not self.queue_path.exists():
            return []
        out: list[dict] = []
        with self.queue_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

    def _rewrite(self, remaining: list[dict]) -> None:
        """Atomically replace the queue with the not-yet-delivered items."""
        tmp = self.queue_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for item in remaining:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            fh.flush()
        tmp.replace(self.queue_path)

    def flush(self, sender: Sender) -> int:
        """Attempt delivery of all pending detections.

        Stops at the first failure (assumed offline) and keeps that item plus the
        rest queued. Returns the number successfully delivered.
        """
        pending = self.pending()
        delivered = 0
        for i, detection in enumerate(pending):
            try:
                ok = sender(detection)
            except Exception:  # network error => treat as offline, stop here.
                ok = False
            if not ok:
                self._rewrite(pending[i:])
                return delivered
            delivered += 1
        self._rewrite([])  # all delivered
        return delivered


def http_sender(api_url: str, timeout: float = 10.0) -> Sender:
    """Build a urllib-based sender that POSTs to ``/api/v1/detections``."""
    endpoint = api_url.rstrip("/") + "/api/v1/detections"

    def _send(detection: dict) -> bool:
        data = json.dumps(detection).encode("utf-8")
        req = urllib.request.Request(
            endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
                return 200 <= resp.status < 300
        except (urllib.error.URLError, TimeoutError):
            return False

    return _send
