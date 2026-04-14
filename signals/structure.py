from __future__ import annotations

import pandas as pd

from signals.common import atr
from signals.models import StructureSignal


def _confirmed_pivots(frame: pd.DataFrame, left: int = 3, right: int = 3) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for idx in range(left, len(frame) - right):
        window = frame.iloc[idx - left : idx + right + 1]
        current_high = float(frame.iloc[idx]["high"])
        current_low = float(frame.iloc[idx]["low"])
        if current_high >= float(window["high"].max()):
            highs.append((idx, current_high))
        if current_low <= float(window["low"].min()):
            lows.append((idx, current_low))
    return highs, lows


def _latest_fvg(frame: pd.DataFrame) -> tuple[tuple[float, float] | None, bool]:
    latest_zone: tuple[float, float] | None = None
    mitigated = False
    for idx in range(2, len(frame)):
        c1 = frame.iloc[idx - 2]
        c3 = frame.iloc[idx]
        if float(c3["low"]) > float(c1["high"]):
            latest_zone = (float(c1["high"]), float(c3["low"]))
            current_low = float(frame.iloc[-1]["low"])
            current_high = float(frame.iloc[-1]["high"])
            mitigated = current_low <= latest_zone[1] and current_high >= latest_zone[0]
        elif float(c3["high"]) < float(c1["low"]):
            latest_zone = (float(c3["high"]), float(c1["low"]))
            current_low = float(frame.iloc[-1]["low"])
            current_high = float(frame.iloc[-1]["high"])
            mitigated = current_low <= latest_zone[1] and current_high >= latest_zone[0]
    return latest_zone, mitigated


def analyze_market_structure(
    frame: pd.DataFrame,
    *,
    ticker: str,
    market: str,
    timeframe: str,
    pivot_left: int = 3,
    pivot_right: int = 3,
    equal_tolerance_pct: float = 0.2,
) -> StructureSignal:
    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    if working.empty or len(working) < pivot_left + pivot_right + 10:
        return StructureSignal(
            ticker=ticker,
            market=market,
            timeframe=timeframe,
            direction="neutral",
            structure_type="insufficient_data",
            key_levels={},
            explanation="Not enough candles to confirm market structure.",
        )

    highs, lows = _confirmed_pivots(working, pivot_left, pivot_right)
    latest_close = float(working.iloc[-1]["close"])
    atr_series = atr(working, 14)
    atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    previous_high = highs[-2][1] if len(highs) >= 2 else None
    previous_low = lows[-2][1] if len(lows) >= 2 else None
    current_swing_high = highs[-1][1] if highs else None
    current_swing_low = lows[-1][1] if lows else None

    bullish_bos = previous_high is not None and latest_close > previous_high
    bearish_bos = previous_low is not None and latest_close < previous_low
    choch = False
    if len(highs) >= 2 and len(lows) >= 2:
        choch = (highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1] and bullish_bos) or (
            highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1] and bearish_bos
        )
    equal_highs = bool(len(highs) >= 2 and previous_high and abs(highs[-1][1] - previous_high) / previous_high * 100 <= equal_tolerance_pct)
    equal_lows = bool(len(lows) >= 2 and previous_low and abs(lows[-1][1] - previous_low) / previous_low * 100 <= equal_tolerance_pct)
    liquidity_sweep = bool(
        previous_high is not None and float(working.iloc[-1]["high"]) > previous_high and latest_close < previous_high
    ) or bool(
        previous_low is not None and float(working.iloc[-1]["low"]) < previous_low and latest_close > previous_low
    )
    inducement = bool(len(lows) >= 3 and previous_low is not None and lows[-1][1] > lows[-2][1] > lows[-3][1])

    direction = "bullish" if bullish_bos else "bearish" if bearish_bos else "neutral"
    structure_type = "BOS" if bullish_bos or bearish_bos else "CHOCH" if choch else "range"

    order_block_zone = None
    if bullish_bos:
        prior_red = working.iloc[:-1].loc[working.iloc[:-1]["close"] < working.iloc[:-1]["open"]]
        if not prior_red.empty:
            candle = prior_red.iloc[-1]
            order_block_zone = (float(candle["low"]), float(candle["high"]))
    elif bearish_bos:
        prior_green = working.iloc[:-1].loc[working.iloc[:-1]["close"] > working.iloc[:-1]["open"]]
        if not prior_green.empty:
            candle = prior_green.iloc[-1]
            order_block_zone = (float(candle["low"]), float(candle["high"]))

    fvg_zone, fvg_mitigated = _latest_fvg(working)
    bos_level = previous_high if bullish_bos else previous_low if bearish_bos else current_swing_high or current_swing_low
    retest_zone = None
    if bos_level is not None and atr_value:
        retest_zone = (bos_level - atr_value * 0.25, bos_level + atr_value * 0.25)

    explanation_bits = [
        f"Latest confirmed structure is {structure_type} with {direction} bias.",
        "Liquidity sweep detected." if liquidity_sweep else "No fresh sweep on the latest bar.",
        "Fresh FVG available." if fvg_zone and not fvg_mitigated else "No fresh unmitigated FVG.",
    ]
    return StructureSignal(
        ticker=ticker,
        market=market,
        timeframe=timeframe,
        direction=direction,
        structure_type=structure_type,
        key_levels={
            "previous_high": previous_high,
            "previous_low": previous_low,
            "current_swing_high": current_swing_high,
            "current_swing_low": current_swing_low,
            "bos_level": bos_level,
        },
        retest_zone=retest_zone,
        invalidation_level=current_swing_low if direction == "bullish" else current_swing_high,
        bos=bullish_bos or bearish_bos,
        choch=choch,
        liquidity_sweep=liquidity_sweep,
        equal_highs=equal_highs,
        equal_lows=equal_lows,
        inducement=inducement,
        order_block_zone=order_block_zone,
        fvg_zone=fvg_zone,
        fvg_mitigated=fvg_mitigated,
        explanation=" ".join(explanation_bits),
    )
