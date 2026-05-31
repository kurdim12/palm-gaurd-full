"""API route tests against the in-memory backend."""

from __future__ import annotations


def test_health_reports_in_memory(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["db"] == "in-memory"


def test_detection_roundtrip_creates_tree_and_detection(client):
    payload = {
        "device_id": "pi-1",
        "tree_id": "tree-xyz",
        "label": "infested",
        "confidence": 0.92,
    }
    res = client.post("/api/v1/detections", json=payload)
    assert res.status_code == 201
    body = res.json()
    assert body["detection"]["tree_id"] == "tree-xyz"
    assert body["tree"]["status"] == "suspect"  # one hit -> suspect, not infested
    assert body["alert"] is None

    listed = client.get("/api/v1/trees/tree-xyz/detections").json()
    assert len(listed) == 1
    assert listed[0]["confidence"] == 0.92


def test_three_confident_hits_flip_and_alert(client):
    payload = {
        "device_id": "pi-1",
        "tree_id": "tree-flip",
        "label": "infested",
        "confidence": 0.9,
    }
    last = None
    for _ in range(3):
        last = client.post("/api/v1/detections", json=payload).json()
    assert last["tree"]["status"] == "infested"
    assert last["status_changed"] is True
    assert last["alert"] is not None
    assert "سوسة" in last["alert"]["message_ar"]  # Arabic-first alert

    open_alerts = client.get("/api/v1/alerts?only_open=true").json()
    assert len(open_alerts) == 1


def test_alert_dedupe_within_rate_limit(client):
    payload = {
        "device_id": "pi-1",
        "tree_id": "tree-spam",
        "label": "infested",
        "confidence": 0.9,
    }
    for _ in range(6):  # well past the streak
        client.post("/api/v1/detections", json=payload)
    # Despite many infested hits, only one alert should exist (deduped).
    alerts = client.get("/api/v1/alerts").json()
    assert len([a for a in alerts if a["tree_id"] == "tree-spam"]) == 1


def test_mark_treated(client):
    client.post(
        "/api/v1/detections",
        json={"device_id": "d", "tree_id": "t-treat", "label": "infested", "confidence": 0.9},
    )
    res = client.post("/api/v1/trees/t-treat/treat")
    assert res.status_code == 200
    assert res.json()["status"] == "treated"


def test_acknowledge_alert(client):
    for _ in range(3):
        client.post(
            "/api/v1/detections",
            json={"device_id": "d", "tree_id": "t-ack", "label": "infested", "confidence": 0.9},
        )
    alert = client.get("/api/v1/alerts").json()[0]
    res = client.post(f"/api/v1/alerts/{alert['id']}/acknowledge")
    assert res.status_code == 200
    assert res.json()["acknowledged"] is True
    assert client.get("/api/v1/alerts?only_open=true").json() == []


def test_status_counts(client):
    client.post(
        "/api/v1/detections",
        json={"device_id": "d", "tree_id": "t-c", "label": "infested", "confidence": 0.9},
    )
    counts = client.get("/api/v1/status/counts").json()
    assert counts["suspect"] == 1
