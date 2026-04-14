from __future__ import annotations

import numpy as np
import pandas as pd

from data.symbols import benchmark_for_market
from signals.common import atr, efficiency_ratio, ema, obv, rolling_percentile
from signals.models import RegimeSignal, RelativeStrengthSignal, VolumeSignal


def analyze_volume_participation(frame: pd.DataFrame, breakout_level: float | None = None) -> VolumeSignal:
    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    latest = working.iloc[-1]
    avg20 = working["volume"].tail(21).iloc[:-1].mean() if len(working) >= 21 else working["volume"].mean()
    volume_ratio = float(latest["volume"] / avg20) if avg20 else 1.0
    obv_series = obv(working)
    obv_confirmation = len(obv_series) > 5 and float(obv_series.iloc[-1]) >= float(obv_series.iloc[-5])
    typical_price = (working["high"] + working["low"] + working["close"]) / 3.0
    rolling_vwap = (typical_price * working["volume"]).rolling(20, min_periods=5).sum() / working["volume"].rolling(20, min_periods=5).sum()
    vwap_alignment = bool(pd.notna(rolling_vwap.iloc[-1]) and float(latest["close"]) >= float(rolling_vwap.iloc[-1]))
    anchor_idx = max(len(working) - 20, 0)
    if breakout_level is not None:
        matches = working.index[working["close"] >= breakout_level].tolist()
        if matches:
            anchor_idx = matches[0]
    anchored_slice = working.iloc[anchor_idx:]
    anchored_typical = (anchored_slice["high"] + anchored_slice["low"] + anchored_slice["close"]) / 3.0
    anchored_vwap = float((anchored_typical * anchored_slice["volume"]).sum() / anchored_slice["volume"].sum()) if anchored_slice["volume"].sum() else None
    dry_up = len(working) >= 11 and float(working["volume"].tail(5).mean()) < float(working["volume"].tail(20).mean()) * 0.85

    penalty_flags: list[str] = []
    if volume_ratio < 1.2:
        penalty_flags.append("breakout_without_participation")
    candle_spread = float(latest["high"] - latest["low"])
    atr_series = atr(working, 14)
    atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    if atr_value and candle_spread > atr_value * 1.5 and volume_ratio < 1.3:
        penalty_flags.append("wide_spread_weak_volume")
    if volume_ratio > 3.0 and float(latest["close"]) < float(latest["high"]) * 0.98:
        penalty_flags.append("possible_exhaustion")

    return VolumeSignal(
        volume_ratio=round(volume_ratio, 2),
        relative_volume=round(volume_ratio, 2),
        obv_confirmation=obv_confirmation,
        vwap_alignment=vwap_alignment,
        anchored_vwap=round(anchored_vwap, 4) if anchored_vwap is not None else None,
        dry_up_before_expansion=dry_up,
        penalty_flags=penalty_flags,
    )


def analyze_relative_strength(
    frame: pd.DataFrame,
    benchmark_frame: pd.DataFrame,
    *,
    benchmark_symbol: str,
    sector_frame: pd.DataFrame | None = None,
    sector_symbol: str | None = None,
) -> RelativeStrengthSignal:
    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    benchmark = benchmark_frame.copy().sort_values("datetime").reset_index(drop=True)
    joined = pd.merge(
        working[["datetime", "close"]],
        benchmark[["datetime", "close"]],
        on="datetime",
        how="inner",
        suffixes=("_asset", "_benchmark"),
    )
    if joined.empty:
        return RelativeStrengthSignal(
            benchmark_symbol=benchmark_symbol,
            sector_benchmark_symbol=sector_symbol,
            score=0.0,
            trend_persistence=0.0,
            smoothness=0.0,
            one_week_alpha=0.0,
            one_month_alpha=0.0,
            three_month_alpha=0.0,
            explanation="Benchmark overlap was unavailable.",
        )

    def _alpha(window: int) -> float:
        if len(joined) < window + 1:
            return 0.0
        asset_return = float(joined["close_asset"].iloc[-1] / joined["close_asset"].iloc[-window - 1] - 1.0)
        benchmark_return = float(joined["close_benchmark"].iloc[-1] / joined["close_benchmark"].iloc[-window - 1] - 1.0)
        return asset_return - benchmark_return

    one_week = _alpha(5)
    one_month = _alpha(21)
    three_month = _alpha(63)
    rs_line = joined["close_asset"] / joined["close_benchmark"]
    smoothness = efficiency_ratio(rs_line, 20)
    persistence = float((rs_line.diff().tail(20) > 0).mean())
    raw_score = ((one_week * 20) + (one_month * 35) + (three_month * 45) + (smoothness * 20) + (persistence * 10)) * 10
    score = max(0.0, min(100.0, raw_score))
    return RelativeStrengthSignal(
        benchmark_symbol=benchmark_symbol,
        sector_benchmark_symbol=sector_symbol,
        score=round(score, 2),
        trend_persistence=round(persistence * 100.0, 2),
        smoothness=round(smoothness * 100.0, 2),
        one_week_alpha=round(one_week * 100.0, 2),
        one_month_alpha=round(one_month * 100.0, 2),
        three_month_alpha=round(three_month * 100.0, 2),
        explanation="Relative strength is measured against the selected benchmark with 1W/1M/3M alpha and path smoothness.",
    )


def analyze_market_regime(benchmark_frame: pd.DataFrame, *, market: str, benchmark_symbol: str) -> RegimeSignal:
    working = benchmark_frame.copy().sort_values("datetime").reset_index(drop=True)
    if working.empty or len(working) < 220:
        return RegimeSignal(
            market=market,
            benchmark_symbol=benchmark_symbol,
            direction="neutral",
            trend_strength=0.0,
            volatility_state="unknown",
            breadth_like_proxy=0.0,
            explanation="Insufficient benchmark history.",
        )
    working["ema20"] = ema(working["close"], 20)
    working["ema50"] = ema(working["close"], 50)
    working["ema200"] = ema(working["close"], 200)
    working["atr14"] = atr(working, 14)
    working["atr_pctile"] = rolling_percentile(working["atr14"], 252)
    latest = working.iloc[-1]
    bullish = float(latest["close"]) > float(latest["ema20"]) > float(latest["ema50"]) > float(latest["ema200"])
    bearish = float(latest["close"]) < float(latest["ema20"]) < float(latest["ema50"]) < float(latest["ema200"])
    direction = "bullish" if bullish else "bearish" if bearish else "neutral"
    trend_strength = float(
        np.mean(
            [
                float(float(latest["close"]) > float(latest["ema20"])),
                float(float(latest["ema20"]) > float(latest["ema50"])),
                float(float(latest["ema50"]) > float(latest["ema200"])),
            ],
        )
        * 100.0,
    )
    atr_pctile = float(latest["atr_pctile"]) if pd.notna(latest["atr_pctile"]) else 0.5
    volatility_state = "expanding" if atr_pctile > 0.65 else "compressed" if atr_pctile < 0.35 else "normal"
    breadth_proxy = float((working["close"].tail(50) > working["ema20"].tail(50)).mean() * 100.0)
    return RegimeSignal(
        market=market,
        benchmark_symbol=benchmark_symbol,
        direction=direction,
        trend_strength=round(trend_strength, 2),
        volatility_state=volatility_state,
        breadth_like_proxy=round(breadth_proxy, 2),
        explanation=f"{market} regime is {direction} with {volatility_state} volatility relative to {benchmark_symbol}.",
    )


def benchmark_symbol_for_market(market: str, benchmark_map: dict[str, dict[str, str]]) -> str:
    return benchmark_for_market(market, benchmark_map)
