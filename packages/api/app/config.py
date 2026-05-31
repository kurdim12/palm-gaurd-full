"""API settings, loaded from the environment (.env at repo root).

No Supabase env => the in-memory DB fallback is used, so the API runs with zero
external services configured (CLAUDE.md: "Runs with no DB configured").
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Backend configuration."""

    model_config = SettingsConfigDict(env_file=(".env", "../../.env"), extra="ignore")

    supabase_url: str = ""
    supabase_service_key: str = ""

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Status engine debounce (see status_engine.py).
    status_infested_streak: int = 3
    status_confidence_threshold: float = 0.6

    # Alerts.
    alert_provider: str = "log"  # none | log | twilio | whatsapp
    alert_rate_limit_minutes: int = 60
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from: str = ""
    alert_to: str = ""

    @property
    def use_supabase(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
