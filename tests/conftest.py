from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def bullish_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=320, freq="B", tz="UTC")
    trend = np.linspace(100, 180, len(dates))
    noise = np.sin(np.arange(len(dates)) / 12.0) * 1.8
    close = trend + noise
    open_ = close - 0.8
    high = close + 1.2
    low = close - 1.4
    volume = np.full(len(dates), 1_000_000.0)
    volume[-1] = 2_500_000.0
    close[-1] = close[:-1].max() * 1.03
    high[-1] = close[-1] * 1.01
    low[-1] = close[-1] * 0.985
    open_[-1] = close[-1] * 0.99
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "symbol": "TEST",
            "market": "US",
            "exchange": "US",
        },
    )


@pytest.fixture()
def benchmark_df() -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=320, freq="B", tz="UTC")
    close = np.linspace(100, 145, len(dates))
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": close - 0.4,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.full(len(dates), 5_000_000.0),
            "symbol": "SPY",
            "market": "US",
            "exchange": "US",
        },
    )
