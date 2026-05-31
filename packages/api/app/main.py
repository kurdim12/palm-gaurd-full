"""Palm Guard FastAPI application.

Runs with no external services: if Supabase env is unset it uses the in-memory
backend (CLAUDE.md). CORS is open for the local dashboard during development.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import alerts, detections, trees

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Palm Guard API",
    version="0.1.0",
    description="Acoustic Red Palm Weevil detection backend.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detections.router)
app.include_router(trees.router)
app.include_router(alerts.router)


@app.get("/health", tags=["meta"])
def health() -> dict:
    """Liveness + which DB backend is active."""
    settings = get_settings()
    return {
        "status": "ok",
        "db": "supabase" if settings.use_supabase else "in-memory",
        "version": app.version,
    }
