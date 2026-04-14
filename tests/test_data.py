from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import load_app_config
from data.loaders import DataEngine
from data.providers import BaseMarketDataProvider
from data.symbols import SymbolRecord, benchmark_for_market, normalize_symbol, parse_manual_watchlist
from stopick_app.workstation import _timeframes_for_scan


class StubProvider(BaseMarketDataProvider):
    def fetch_ohlcv(self, symbol: str, interval: str, *, start=None, end=None, period=None) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=10, freq="B", tz="UTC")
        return pd.DataFrame(
            {
                "datetime": dates,
                "open": [1.0] * 10,
                "high": [1.2] * 10,
                "low": [0.9] * 10,
                "close": [1.1] * 10,
                "volume": [100] * 10,
                "symbol": ["TEST"] * 10,
            },
        )

    def fetch_corporate_calendar(self, symbols):
        return pd.DataFrame(columns=["symbol", "event_type", "event_date"])


def test_symbol_helpers_and_loader_cache() -> None:
    assert normalize_symbol("reliance", "NSE") == "RELIANCE.NS"
    assert parse_manual_watchlist("AAPL, MSFT", "US") == ["AAPL", "MSFT"]
    assert _timeframes_for_scan("1d") == ["1d"]
    assert _timeframes_for_scan("4h") == ["1d", "4h"]

    config = load_app_config(Path(__file__).resolve().parent.parent)
    engine = DataEngine(config, provider=StubProvider())
    record = SymbolRecord(symbol="TEST", market="US", exchange="NASDAQ")
    frame = engine.fetch_symbol(record, "1d", refresh=True)
    assert not frame.empty
    assert benchmark_for_market("US", config.data.benchmark_map) == "SPY"
