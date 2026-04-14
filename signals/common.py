from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=period, adjust=False).mean()


def atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = pd.to_numeric(series, errors="coerce").diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_values = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_values
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr_values
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def obv(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def rolling_percentile(series: pd.Series, window: int = 252) -> pd.Series:
    def _rank(values: np.ndarray) -> float:
        if len(values) <= 1:
            return 0.5
        return float(pd.Series(values).rank(pct=True).iloc[-1])

    return pd.to_numeric(series, errors="coerce").rolling(window, min_periods=max(20, window // 5)).apply(_rank, raw=True)


def linreg_slope(series: pd.Series, window: int = 20) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna().tail(window)
    if len(cleaned) < 5:
        return 0.0
    x = np.arange(len(cleaned))
    slope = np.polyfit(x, cleaned.to_numpy(dtype=float), 1)[0]
    return float(slope)


def efficiency_ratio(series: pd.Series, window: int = 20) -> float:
    cleaned = pd.to_numeric(series, errors="coerce").dropna().tail(window)
    if len(cleaned) < 5:
        return 0.0
    net_change = abs(cleaned.iloc[-1] - cleaned.iloc[0])
    path = cleaned.diff().abs().sum()
    return float(net_change / path) if path else 0.0
