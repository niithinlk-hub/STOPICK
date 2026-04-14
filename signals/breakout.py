from __future__ import annotations

import math
from typing import Any

import pandas as pd

from signals.common import atr, ema
from signals.models import BreakoutSignal, StructureSignal, TrendSignal


def _candle_close_quality(row: pd.Series) -> float:
    total_range = float(row["high"]) - float(row["low"])
    if total_range <= 0:
        return 0.0
    body = abs(float(row["close"]) - float(row["open"]))
    close_near_high = 1.0 - ((float(row["high"]) - float(row["close"])) / total_range)
    return max(0.0, min(1.0, ((body / total_range) * 0.55) + (close_near_high * 0.45)))


def _tightness_score(frame: pd.DataFrame, lookback: int = 20) -> float:
    recent = frame.tail(lookback)
    if recent.empty:
        return 0.0
    compression = recent["close"].std() / max(recent["close"].mean(), 1e-9)
    return round(max(0.0, min(1.0, 1.0 - (compression * 8))), 4)


def _volume_ratio(frame: pd.DataFrame) -> float:
    recent = frame.tail(21)
    if len(recent) < 21:
        return 1.0
    average = recent["volume"].iloc[:-1].mean()
    return float(recent["volume"].iloc[-1] / average) if average else 1.0


def _distance_pct(current: float, level: float | None) -> float | None:
    if level in {None, 0}:
        return None
    return round(((current / level) - 1.0) * 100.0, 4)


def _overhead_resistance_pct(frame: pd.DataFrame, current_price: float) -> float | None:
    highs = frame["high"].tail(252)
    if highs.empty:
        return None
    overhead = highs.max()
    if overhead <= current_price:
        return None
    return round(((overhead / current_price) - 1.0) * 100.0, 4)


def _base_breakout(frame: pd.DataFrame, buffer_pct: float, lookback: int) -> dict[str, Any]:
    if len(frame) < lookback + 5:
        return {}
    base = frame.iloc[:-1].tail(lookback)
    breakout_level = float(base["high"].max())
    last = frame.iloc[-1]
    buffered = breakout_level * (1 + buffer_pct / 100.0)
    return {
        "pattern_name": "Base breakout",
        "is_valid": float(last["close"]) > buffered,
        "breakout_level": breakout_level,
        "buffered_level": buffered,
        "invalidation_level": float(base["low"].min()),
        "touches": int((base["high"] >= breakout_level * 0.995).sum()),
    }


def _ath_breakout(frame: pd.DataFrame, buffer_pct: float) -> dict[str, Any]:
    if len(frame) < 260:
        return {}
    previous_high = float(frame.iloc[:-1].tail(252)["high"].max())
    last = frame.iloc[-1]
    buffered = previous_high * (1 + buffer_pct / 100.0)
    return {
        "pattern_name": "ATH breakout",
        "is_valid": float(last["close"]) > buffered,
        "breakout_level": previous_high,
        "buffered_level": buffered,
        "invalidation_level": float(frame.tail(20)["low"].min()),
        "touches": 1,
    }


def _flag_breakout(frame: pd.DataFrame, buffer_pct: float) -> dict[str, Any]:
    if len(frame) < 35:
        return {}
    impulse = frame.iloc[-35:-15]
    flag = frame.iloc[-15:-1]
    if impulse.empty or flag.empty:
        return {}
    impulse_gain = float(impulse["close"].iloc[-1] / impulse["close"].iloc[0] - 1.0)
    flag_depth = float((flag["high"].max() - flag["low"].min()) / max(impulse["close"].iloc[-1], 1e-9))
    breakout_level = float(flag["high"].max())
    last_close = float(frame.iloc[-1]["close"])
    buffered = breakout_level * (1 + buffer_pct / 100.0)
    return {
        "pattern_name": "Flag continuation",
        "is_valid": impulse_gain > 0.08 and flag_depth < 0.08 and last_close > buffered,
        "breakout_level": breakout_level,
        "buffered_level": buffered,
        "invalidation_level": float(flag["low"].min()),
        "touches": 2,
    }


def _cup_handle(frame: pd.DataFrame, buffer_pct: float) -> dict[str, Any]:
    if len(frame) < 80:
        return {}
    cup = frame.iloc[-80:-15]
    handle = frame.iloc[-15:-1]
    if cup.empty or handle.empty:
        return {}
    rim = max(float(cup["high"].iloc[0]), float(cup["high"].iloc[-1]))
    trough = float(cup["low"].min())
    recovery_ratio = (float(cup["close"].iloc[-1]) - trough) / max(rim - trough, 1e-9)
    handle_depth = (float(handle["high"].max()) - float(handle["low"].min())) / max(rim, 1e-9)
    buffered = rim * (1 + buffer_pct / 100.0)
    return {
        "pattern_name": "Cup-and-handle approximation",
        "is_valid": recovery_ratio > 0.8 and handle_depth < 0.08 and float(frame.iloc[-1]["close"]) > buffered,
        "breakout_level": rim,
        "buffered_level": buffered,
        "invalidation_level": float(handle["low"].min()),
        "touches": 2,
    }


def _vcp_breakout(frame: pd.DataFrame, buffer_pct: float, lookback: int) -> dict[str, Any]:
    if len(frame) < lookback + 10:
        return {}
    recent = frame.iloc[-lookback - 1 : -1]
    ranges = (recent["high"] - recent["low"]).rolling(10, min_periods=3).mean().dropna()
    breakout_level = float(recent["high"].max())
    contraction = len(ranges) >= 2 and float(ranges.iloc[-1]) < float(ranges.iloc[0]) * 0.7
    buffered = breakout_level * (1 + buffer_pct / 100.0)
    return {
        "pattern_name": "Volatility contraction breakout",
        "is_valid": contraction and float(frame.iloc[-1]["close"]) > buffered,
        "breakout_level": breakout_level,
        "buffered_level": buffered,
        "invalidation_level": float(recent["low"].tail(10).min()),
        "touches": int((recent["high"] >= breakout_level * 0.995).sum()),
    }


def _event_continuation(frame: pd.DataFrame, event_proximity_days: int | None, market: str, buffer_pct: float) -> dict[str, Any]:
    if event_proximity_days is None:
        return {}
    recent = frame.tail(10)
    if recent.empty:
        return {}
    breakout_level = float(recent.iloc[:-1]["high"].max())
    buffered = breakout_level * (1 + buffer_pct / 100.0)
    label = "Earnings breakout continuation" if market == "US" else "Results breakout continuation"
    return {
        "pattern_name": label,
        "is_valid": event_proximity_days <= 5 and float(frame.iloc[-1]["close"]) > buffered,
        "breakout_level": breakout_level,
        "buffered_level": buffered,
        "invalidation_level": float(recent["low"].min()),
        "touches": 1,
    }


def find_best_breakout(
    frame: pd.DataFrame,
    *,
    market: str,
    trend_signal: TrendSignal,
    structure_signal: StructureSignal,
    relative_strength_score: float,
    breakout_buffer_pct: float = 0.5,
    lookback: int = 40,
    event_proximity_days: int | None = None,
) -> BreakoutSignal:
    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    working["ema20"] = ema(working["close"], 20)
    working["atr14"] = atr(working, 14)
    last = working.iloc[-1]
    candidates = [
        _base_breakout(working, breakout_buffer_pct, lookback),
        _ath_breakout(working, breakout_buffer_pct),
        _flag_breakout(working, breakout_buffer_pct),
        _cup_handle(working, breakout_buffer_pct),
        _vcp_breakout(working, breakout_buffer_pct, min(30, lookback)),
        _event_continuation(working, event_proximity_days, market, breakout_buffer_pct),
    ]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return BreakoutSignal(
            is_valid=False,
            pattern_name="None",
            direction=trend_signal.direction,
            breakout_level=None,
            buffered_level=None,
            current_price=float(last["close"]),
            distance_pct=None,
            candle_quality=0.0,
            tightness_score=0.0,
            volume_expansion=1.0,
            overhead_resistance_pct=None,
            invalidation_level=None,
            explanation="No breakout candidate pattern was available.",
        )

    volume_ratio = _volume_ratio(working)
    candle_quality = _candle_close_quality(last)
    tightness = _tightness_score(working, 20)
    overhead = _overhead_resistance_pct(working, float(last["close"]))
    best = max(
        candidates,
        key=lambda item: (
            float(item.get("is_valid", False)),
            item.get("touches", 0),
            volume_ratio,
            candle_quality,
            tightness,
        ),
    )
    is_valid = bool(best.get("is_valid")) and trend_signal.direction != "bearish"
    reasons = [
        f"{best['pattern_name']} uses breakout level {best['breakout_level']:.2f}.",
        f"Volume ratio is {volume_ratio:.2f}x and candle quality is {candle_quality:.2f}.",
        f"Relative strength score is {relative_strength_score:.1f}.",
        structure_signal.explanation,
    ]
    if overhead is not None and overhead < 3:
        reasons.append("There is nearby higher-timeframe resistance overhead.")
        is_valid = False
    if volume_ratio < 1.2 or candle_quality < 0.55:
        reasons.append("Breakout quality is downgraded by weak participation or close quality.")
    invalidation = best.get("invalidation_level")
    if invalidation is None and pd.notna(last["atr14"]):
        invalidation = float(last["close"]) - float(last["atr14"]) * 1.5

    return BreakoutSignal(
        is_valid=is_valid,
        pattern_name=str(best["pattern_name"]),
        direction=trend_signal.direction,
        breakout_level=float(best["breakout_level"]) if best.get("breakout_level") is not None else None,
        buffered_level=float(best["buffered_level"]) if best.get("buffered_level") is not None else None,
        current_price=float(last["close"]),
        distance_pct=_distance_pct(float(last["close"]), best.get("breakout_level")),
        candle_quality=round(candle_quality * 100.0, 2),
        tightness_score=round(tightness * 100.0, 2),
        volume_expansion=round(volume_ratio, 2),
        overhead_resistance_pct=overhead,
        invalidation_level=float(invalidation) if invalidation is not None and not math.isnan(float(invalidation)) else None,
        explanation=" ".join(reasons),
        metrics={
            "touches": best.get("touches", 0),
            "relative_strength_score": relative_strength_score,
            "trend_direction": trend_signal.direction,
            "bos_present": structure_signal.bos,
        },
    )
