"""Alert dispatch with bilingual templates, rate-limiting, and dedupe.

One infested tree must not spam: a new alert for a tree is suppressed if an alert
was already raised within ``rate_limit_minutes``. Templates are AR + EN
(Arabic-first). The transport is pluggable; ``log`` is the safe default and
``twilio`` / ``whatsapp`` wire a real provider when env keys are present.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from .config import Settings
from .db.base import Database
from .models import Alert, Tree

logger = logging.getLogger("palmguard.alerts")


def render_messages(tree: Tree) -> tuple[str, str]:
    """Return ``(message_ar, message_en)`` for an infestation alert."""
    name_ar = tree.name_ar or tree.id
    name_en = tree.name_en or tree.id
    ar = (
        f"🚨 تنبيه سوسة النخيل الحمراء: تم رصد إصابة محتملة في النخلة {name_ar}. "
        f"يرجى الفحص العاجل."
    )
    en = (
        f"🚨 Red Palm Weevil alert: likely infestation detected on tree {name_en}. "
        f"Please inspect urgently."
    )
    return ar, en


class AlertDispatcher:
    """Decides whether to raise an alert and sends it via the configured provider."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db

    def maybe_alert(self, tree: Tree) -> Alert | None:
        """Create + send an alert for a newly-infested tree, unless rate-limited."""
        if self._rate_limited(tree):
            logger.info("Alert for %s suppressed (rate limit / dedupe).", tree.id)
            return None

        message_ar, message_en = render_messages(tree)
        alert = Alert(tree_id=tree.id, message_ar=message_ar, message_en=message_en)
        alert = self.db.add_alert(alert)
        self._send(alert)
        return alert

    def _rate_limited(self, tree: Tree) -> bool:
        last = self.db.latest_alert_for_tree(tree.id)
        if last is None:
            return False
        window = timedelta(minutes=self.settings.alert_rate_limit_minutes)
        return (tree.updated_at - last.created_at) < window

    def _send(self, alert: Alert) -> None:
        provider = self.settings.alert_provider
        if provider in ("none", ""):
            return
        if provider == "log":
            logger.warning("ALERT %s | %s", alert.tree_id, alert.message_en)
            return
        if provider in ("twilio", "whatsapp"):
            self._send_twilio(alert, whatsapp=provider == "whatsapp")
            return
        logger.error("Unknown alert provider %r; alert not sent.", provider)

    def _send_twilio(self, alert: Alert, whatsapp: bool) -> None:
        """Send via Twilio SMS or WhatsApp. No-ops with a warning if unconfigured."""
        s = self.settings
        if not (s.twilio_account_sid and s.twilio_auth_token and s.twilio_from and s.alert_to):
            logger.warning("Twilio not fully configured; alert %s not sent.", alert.id)
            return
        try:
            from twilio.rest import Client  # noqa: PLC0415 — optional dep
        except ImportError:
            logger.error("twilio package not installed; cannot send alert %s.", alert.id)
            return

        client = Client(s.twilio_account_sid, s.twilio_auth_token)
        prefix = "whatsapp:" if whatsapp else ""
        body = f"{alert.message_ar}\n\n{alert.message_en}"
        client.messages.create(
            from_=f"{prefix}{s.twilio_from}", to=f"{prefix}{s.alert_to}", body=body
        )
        logger.info("Alert %s dispatched via %s.", alert.id, "whatsapp" if whatsapp else "sms")
