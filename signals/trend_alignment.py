from __future__ import annotations

import pandas as pd

from signals.common import adx, ema, linreg_slope
from signals.models import TrendSignal


def _hh_hl_score(frame: pd.DataFrame) -> float:
    recent = frame.tail(40).copy()
    if len(recent) < 20:
        return 0.0
    highs = recent["high"].rolling(5, min_periods=5).max().dropna()
    lows = recent["low"].rolling(5, min_periods=5).min().dropna()
    if len(highs) < 4 or len(lows) < 4:
        return 0.0
    higher_highs = float(highs.iloc[-1] > highs.iloc[-4])
    higher_lows = float(lows.iloc[-1] > lows.iloc[-4])
    lower_highs = float(highs.iloc[-1] < highs.iloc[-4])
    lower_lows = float(lows.iloc[-1] < lows.iloc[-4])
    if higher_highs and higher_lows:
        return 1.0
    if lower_highs and lower_lows:
        return -1.0
    return 0.0


def analyze_timeframe_trend(frame: pd.DataFrame, timeframe: str, adx_threshold: float = 20.0) -> tuple[str, float, dict[str, float | str | None]]:
    if frame.empty or len(frame) < 220:
        return "neutral", 0.0, {"timeframe": timeframe}

    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    working["ema20"] = ema(working["close"], 20)
    working["ema50"] = ema(working["close"], 50)
    working["ema200"] = ema(working["close"], 200)
    working["adx14"] = adx(working, 14)

    latest = working.iloc[-1]
    hh_hl_state = _hh_hl_score(working)
    ema20_slope = linreg_slope(working["ema20"], 15)

    bullish_checks = [
        float(latest["ema20"] > latest["ema50"] > latest["ema200"]),
        float(latest["close"] > latest["ema20"] and latest["close"] > latest["ema50"]),
        float(hh_hl_state > 0),
        float((latest["adx14"] or 0) >= adx_threshold),
        float(ema20_slope > 0),
    ]
    bearish_checks = [
        float(latest["ema20"] < latest["ema50"] < latest["ema200"]),
        float(latest["close"] < latest["ema20"] and latest["close"] < latest["ema50"]),
        float(hh_hl_state < 0),
        float((latest["adx14"] or 0) >= adx_threshold),
        float(ema20_slope < 0),
    ]
    bullish_score = sum(bullish_checks) / len(bullish_checks)
    bearish_score = sum(bearish_checks) / len(bearish_checks)
    if bullish_score > bearish_score and bullish_score >= 0.6:
        direction = "bullish"
        score = bullish_score
    elif bearish_score > bullish_score and bearish_score >= 0.6:
        direction = "bearish"
        score = bearish_score
    else:
        direction = "neutral"
        score = max(bullish_score, bearish_score) * 0.5

    metrics = {
        "timeframe": timeframe,
        "close": float(latest["close"]),
        "ema20": float(latest["ema20"]),
        "ema50": float(latest["ema50"]),
        "ema200": float(latest["ema200"]),
        "adx14": float(latest["adx14"]) if pd.notna(latest["adx14"]) else None,
        "ema20_slope": float(ema20_slope),
        "hh_hl_state": float(hh_hl_state),
    }
    return direction, float(score), metrics


def analyze_trend_alignment(
    frame_map: dict[str, pd.DataFrame],
    *,
    adx_threshold: float = 20.0,
    preferred_timeframes: list[str] | None = None,
) -> TrendSignal:
    timeframes = preferred_timeframes or ["1d", "4h", "1h", "15m"]
    scores: dict[str, float] = {}
    metrics: dict[str, float | str | None] = {}
    bullish_votes = 0
    bearish_votes = 0

    for timeframe in timeframes:
        frame = frame_map.get(timeframe)
        if frame is None or frame.empty:
            continue
        direction, score, timeframe_metrics = analyze_timeframe_trend(frame, timeframe, adx_threshold)
        scores[timeframe] = score
        metrics.update({f"{timeframe}_{key}": value for key, value in timeframe_metrics.items()})
        if direction == "bullish":
            bullish_votes += 1
        elif direction == "bearish":
            bearish_votes += 1

    if bullish_votes > bearish_votes and scores:
        direction = "bullish"
    elif bearish_votes > bullish_votes and scores:
        direction = "bearish"
    else:
        direction = "neutral"

    alignment_confidence = (max(bullish_votes, bearish_votes) / max(len(scores), 1)) * 100.0
    strength_score = (sum(scores.values()) / max(len(scores), 1)) * 100.0
    return TrendSignal(
        direction=direction,
        strength_score=round(strength_score, 2),
        alignment_confidence=round(alignment_confidence, 2),
        timeframe_scores={key: round(value * 100.0, 2) for key, value in scores.items()},
        metrics=metrics,
    )
