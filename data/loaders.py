from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

from config import AppConfig
from data.cache import DiskCache
from data.providers import BaseMarketDataProvider, YFinanceProvider
from data.symbols import SymbolRecord


def _start_for_interval(interval: str, lookback_bars: int) -> datetime:
    now = datetime.now(timezone.utc)
    if interval == "1d":
        return now - timedelta(days=max(lookback_bars * 2, 365))
    if interval == "1h":
        return now - timedelta(days=max(lookback_bars // 6, 180))
    if interval == "15m":
        return now - timedelta(days=max(lookback_bars // 20, 60))
    return now - timedelta(days=max(lookback_bars // 40, 30))


def _resample_intraday(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    if interval == "4h":
        indexed = frame.set_index("datetime").sort_index()
        resampled = indexed.resample("4h").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "symbol": "last"},
        )
        return resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return frame


@dataclass(slots=True)
class DataEngine:
    config: AppConfig
    provider: BaseMarketDataProvider | None = None
    cache: DiskCache = field(init=False)

    def __post_init__(self) -> None:
        self.provider = self.provider or YFinanceProvider()
        self.cache = DiskCache(self.config.project_root / "data_store" / "cache")

    def fetch_symbol(
        self,
        symbol_record: SymbolRecord,
        interval: str,
        *,
        lookback_bars: int = 400,
        refresh: bool = False,
    ) -> pd.DataFrame:
        cache_key = f"{symbol_record.symbol}|{interval}|{lookback_bars}"
        if not refresh:
            cached = self.cache.get_frame("ohlcv", cache_key)
            if cached is not None and not cached.empty:
                return cached

        fetch_interval = "60m" if interval == "4h" else interval
        frame = self.provider.fetch_ohlcv(
            symbol_record.symbol,
            fetch_interval,
            start=_start_for_interval(fetch_interval, lookback_bars),
        )
        frame = _resample_intraday(frame, interval)
        frame["market"] = symbol_record.market
        frame["exchange"] = symbol_record.exchange
        frame["sector"] = symbol_record.sector
        self.cache.set_frame("ohlcv", cache_key, frame)
        return frame

    def fetch_universe(
        self,
        records: Iterable[SymbolRecord],
        interval: str,
        *,
        lookback_bars: int = 400,
        refresh: bool = False,
        max_workers: int = 8,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
        frames: dict[str, pd.DataFrame] = {}
        failures: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(self.fetch_symbol, record, interval, lookback_bars=lookback_bars, refresh=refresh): record
                for record in records
            }
            for future in as_completed(future_map):
                record = future_map[future]
                try:
                    frame = future.result()
                    if frame.empty:
                        failures[record.symbol] = "No data returned."
                    else:
                        frames[record.symbol] = frame
                except Exception as exc:
                    failures[record.symbol] = str(exc)
        return frames, failures
