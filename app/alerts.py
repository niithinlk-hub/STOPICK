from __future__ import annotations

import json
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import requests

from config import AppConfig
from signals.models import SetupSignal


def load_alert_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_alert_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_alert_message(setup: SetupSignal) -> str:
    return (
        f"STOPICK alert\n"
        f"{setup.ticker} ({setup.market})\n"
        f"{setup.setup_family} | {setup.grade} | score {setup.score:.1f}\n"
        f"Pattern: {setup.breakout.pattern_name}\n"
        f"Breakout level: {setup.breakout.breakout_level}\n"
        f"Current price: {setup.breakout.current_price}\n"
        f"Why qualified: {'; '.join(setup.reasons_for[:4])}\n"
        f"Warnings: {'; '.join(setup.risk_warnings[:3]) or 'None'}"
    )


def should_alert(setup: SetupSignal, state: dict[str, Any], min_score: float) -> bool:
    if setup.score < min_score:
        return False
    key = f"{setup.ticker}|{setup.setup_family}|{setup.breakout.breakout_level}"
    return state.get(key) != round(setup.score, 2)


def send_telegram(message: str, config: AppConfig) -> str:
    if not config.alerts.telegram_bot_token or not config.alerts.telegram_chat_id:
        return "Telegram not configured."
    response = requests.post(
        f"https://api.telegram.org/bot{config.alerts.telegram_bot_token}/sendMessage",
        json={"chat_id": config.alerts.telegram_chat_id, "text": message},
        timeout=30,
    )
    response.raise_for_status()
    return "Telegram sent"


def send_email(message: str, subject: str, config: AppConfig) -> str:
    if not all([config.alerts.email_from, config.alerts.email_to, config.alerts.smtp_host, config.alerts.smtp_user, config.alerts.smtp_password]):
        return "Email not configured."
    email = EmailMessage()
    email["From"] = config.alerts.email_from
    email["To"] = config.alerts.email_to
    email["Subject"] = subject
    email.set_content(message)
    with smtplib.SMTP_SSL(config.alerts.smtp_host, 465, timeout=30) as smtp:
        smtp.login(config.alerts.smtp_user, config.alerts.smtp_password)
        smtp.send_message(email)
    return "Email sent"


def send_webhook(payload: dict[str, Any], config: AppConfig) -> str:
    if not config.alerts.webhook_url:
        return "Webhook not configured."
    response = requests.post(config.alerts.webhook_url, json=payload, timeout=30)
    response.raise_for_status()
    return "Webhook sent"
