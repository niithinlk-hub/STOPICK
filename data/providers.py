from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

import pandas as pd

from data.symbols import display_symbol


class BaseMarketDataProvider(ABC):
    @abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def fetch_corporate_calendar(self, symbols: Iterable[str]) -> pd.DataFrame:
        raise NotImplementedError


class YFinanceProvider(BaseMarketDataProvider):
    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
        period: str | None = None,
    ) -> pd.DataFrame:
        import yfinance as yf

        history = yf.download(
            tickers=symbol,
            interval=interval,
            start=start,
            end=end,
            period=period,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if history.empty:
            return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume", "symbol"])

        frame = history.reset_index().rename(
            columns={
                "Date": "datetime",
                "Datetime": "datetime",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            },
        )
        frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame["symbol"] = display_symbol(symbol)
        return frame[["datetime", "open", "high", "low", "close", "volume", "symbol"]].dropna(
            subset=["datetime", "open", "high", "low", "close"],
        )

    def fetch_corporate_calendar(self, symbols: Iterable[str]) -> pd.DataFrame:
        rows: list[dict[str, str | None]] = []
        try:
            import yfinance as yf
        except Exception:
            return pd.DataFrame(columns=["symbol", "event_type", "event_date"])

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                calendar = ticker.calendar
                if calendar is None or calendar.empty:
                    continue
                event_date = None
                if "Earnings Date" in calendar.index and not calendar.loc["Earnings Date"].empty:
                    event_date = pd.to_datetime(calendar.loc["Earnings Date"].iloc[0], utc=True, errors="coerce")
                rows.append({"symbol": display_symbol(symbol), "event_type": "earnings_or_results", "event_date": event_date})
            except Exception:
                continue
        return pd.DataFrame(rows)
