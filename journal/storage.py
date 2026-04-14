from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class JournalStore:
    database_path: Path

    def __post_init__(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    market TEXT NOT NULL,
                    setup_family TEXT NOT NULL,
                    grade TEXT NOT NULL,
                    note TEXT NOT NULL
                )
                """,
            )

    def add_entry(self, created_at: str, ticker: str, market: str, setup_family: str, grade: str, note: str) -> None:
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO trade_journal (created_at, ticker, market, setup_family, grade, note) VALUES (?, ?, ?, ?, ?, ?)",
                (created_at, ticker, market, setup_family, grade, note),
            )

    def load_entries(self) -> pd.DataFrame:
        with sqlite3.connect(self.database_path) as connection:
            return pd.read_sql_query("SELECT * FROM trade_journal ORDER BY created_at DESC", connection)
