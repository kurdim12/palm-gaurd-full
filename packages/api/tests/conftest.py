"""API test fixtures: a TestClient wired to a fresh in-memory DB per test."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import deps  # noqa: E402
from app.db.memory import InMemoryDatabase  # noqa: E402
from app.main import app  # noqa: E402


@pytest.fixture
def db() -> InMemoryDatabase:
    return InMemoryDatabase(seed_demo=False)


@pytest.fixture
def client(db, monkeypatch) -> TestClient:
    # Override the cached DB singleton with a clean per-test instance.
    app.dependency_overrides[deps.get_db] = lambda: db
    # The dispatcher reads the DB via deps.get_db too; patch it to use the test DB.
    monkeypatch.setattr(deps, "get_db", lambda: db)
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
