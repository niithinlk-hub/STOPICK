from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


def _read_secret(key: str) -> str | None:
    value = os.getenv(key)
    if value not in {None, ""}:
        return value
    try:
        import streamlit as st

        secret_value: Any = st.secrets.get(key)
    except Exception:
        return None
    return None if secret_value in {None, ""} else str(secret_value)


class RuntimeConfig(BaseModel):
    app_name: str = "STOPICK"
    default_country: str = "BOTH"
    default_setup_type: str = "breakout"
    min_score_default: int = 75
    cache_ttl_minutes: int = 30
    max_symbols_per_scan: int = 250
    capital_base: float = 1_000_000.0
    risk_per_trade_pct: float = 0.5
    slippage_bps: float = 5.0
    brokerage_bps: float = 3.0
    taxes_bps: float = 2.0


class DataConfig(BaseModel):
    default_provider: str = "yfinance"
    supported_intervals: list[str] = Field(default_factory=lambda: ["1d", "1h", "15m", "5m"])
    benchmark_map: dict[str, dict[str, str]] = Field(
        default_factory=lambda: {
            "NSE": {"broad": "^NSEI", "broad_alt": "^CRSLDX", "tech": "^CNXIT", "bank": "^NSEBANK"},
            "US": {"broad": "SPY", "growth": "QQQ", "semis": "SMH", "financials": "XLF"},
        },
    )
    earnings_window_days: int = 7
    results_window_days: int = 7


class ScoringProfile(BaseModel):
    weights: dict[str, float]
    grade_thresholds: dict[str, float]


class AlertConfig(BaseModel):
    telegram_bot_token: str | None = None
    telegram_chat_id: str | None = None
    email_from: str | None = None
    email_to: str | None = None
    smtp_host: str | None = None
    smtp_user: str | None = None
    smtp_password: str | None = None
    webhook_url: str | None = None


class AppConfig(BaseModel):
    project_root: Path
    runtime: RuntimeConfig
    data: DataConfig
    scoring_profiles: dict[str, ScoringProfile]
    alerts: AlertConfig
    universe_files: dict[str, Path]
    journal_db_path: Path
    alert_state_path: Path


def _load_yaml_config(project_root: Path) -> dict[str, Any]:
    config_path = project_root / "config" / "config.yaml"
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}


@lru_cache(maxsize=1)
def load_app_config(project_root: Path | None = None) -> AppConfig:
    resolved_root = project_root or Path(__file__).resolve().parent.parent
    raw = _load_yaml_config(resolved_root)

    runtime = RuntimeConfig(**raw.get("runtime", {}))
    data = DataConfig(**raw.get("data", {}))
    scoring_profiles = {
        key: ScoringProfile(**value)
        for key, value in (raw.get("scoring_profiles", {}) or {}).items()
    }
    alerts = AlertConfig(
        **{
            **raw.get("alerts", {}),
            "telegram_bot_token": _read_secret("TELEGRAM_BOT_TOKEN"),
            "telegram_chat_id": _read_secret("TELEGRAM_CHAT_ID"),
            "email_from": _read_secret("EMAIL_FROM"),
            "email_to": _read_secret("EMAIL_TO"),
            "smtp_host": _read_secret("SMTP_HOST"),
            "smtp_user": _read_secret("SMTP_USER"),
            "smtp_password": _read_secret("SMTP_PASSWORD"),
            "webhook_url": _read_secret("WEBHOOK_URL"),
        },
    )
    universe_files = {
        name: resolved_root / path
        for name, path in (raw.get("universe_files", {}) or {}).items()
    }

    data_dir = resolved_root / "data_store"
    data_dir.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        project_root=resolved_root,
        runtime=runtime,
        data=data,
        scoring_profiles=scoring_profiles,
        alerts=alerts,
        universe_files=universe_files,
        journal_db_path=Path(
            _read_secret("JOURNAL_DB_PATH") or str(data_dir / "stopick_journal.sqlite"),
        ),
        alert_state_path=Path(
            _read_secret("ALERT_STATE_PATH") or str(data_dir / "stopick_alert_state.json"),
        ),
    )
