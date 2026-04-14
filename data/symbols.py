from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(slots=True)
class SymbolRecord:
    symbol: str
    market: str
    exchange: str
    sector: str = "Unknown"
    market_cap_bucket: str = "Unknown"


def normalize_symbol(symbol: str, market: str) -> str:
    cleaned = str(symbol).strip().upper()
    if not cleaned:
        return ""
    if market == "NSE" and not cleaned.endswith(".NS") and not cleaned.startswith("^"):
        return f"{cleaned}.NS"
    return cleaned


def display_symbol(symbol: str) -> str:
    return symbol[:-3] if symbol.endswith(".NS") else symbol


def parse_manual_watchlist(raw: str, market: str) -> list[str]:
    symbols = [normalize_symbol(item, market) for item in str(raw).replace("\n", ",").split(",")]
    return [symbol for symbol in symbols if symbol]


def load_watchlist_file(path: Path, market: str) -> list[str]:
    if not path.exists():
        return []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [normalize_symbol(line, market) for line in lines if line]


def load_watchlist_csv(path: Path, market_column: str = "market", symbol_column: str = "symbol") -> list[SymbolRecord]:
    frame = pd.read_csv(path)
    rows: list[SymbolRecord] = []
    for _, row in frame.iterrows():
        market = str(row.get(market_column, "US")).upper()
        symbol = normalize_symbol(str(row[symbol_column]), market)
        rows.append(
            SymbolRecord(
                symbol=symbol,
                market=market,
                exchange=str(row.get("exchange", market)),
                sector=str(row.get("sector", "Unknown")),
                market_cap_bucket=str(row.get("market_cap_bucket", "Unknown")),
            ),
        )
    return rows


def benchmark_for_market(market: str, benchmark_map: dict[str, dict[str, str]]) -> str:
    market_key = "NSE" if market == "NSE" else "US"
    return benchmark_map.get(market_key, {}).get("broad", "^NSEI" if market_key == "NSE" else "SPY")
