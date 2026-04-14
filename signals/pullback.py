from __future__ import annotations

import pandas as pd

from signals.common import atr
from signals.models import BreakoutSignal, PullbackSignal, StructureSignal


def find_pullback_entry(
    frame: pd.DataFrame,
    breakout_signal: BreakoutSignal,
    structure_signal: StructureSignal,
) -> PullbackSignal:
    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    if working.empty or breakout_signal.breakout_level is None:
        return PullbackSignal(
            is_valid=False,
            setup_type="none",
            entry_zone=None,
            confirmation_trigger=None,
            stop_zone=None,
            rr_targets={},
            explanation="Breakout level is unavailable for pullback planning.",
        )

    atr_series = atr(working, 14)
    atr_value = float(atr_series.iloc[-1]) if pd.notna(atr_series.iloc[-1]) else 0.0
    latest = working.iloc[-1]
    recent = working.tail(8)
    breakout_level = float(breakout_signal.breakout_level)
    zone_low = breakout_level - atr_value * 0.35
    zone_high = breakout_level + atr_value * 0.2
    price_in_zone = zone_low <= float(latest["close"]) <= zone_high
    dry_up = recent.iloc[:-1]["volume"].mean() > 0 and float(latest["volume"]) <= recent.iloc[:-1]["volume"].mean() * 0.95
    rejection = float(latest["close"]) > float(latest["open"]) and float(latest["low"]) <= breakout_level <= float(latest["high"])
    fvg_fill = bool(structure_signal.fvg_zone and structure_signal.fvg_zone[0] <= float(latest["low"]) <= structure_signal.fvg_zone[1])
    confirmation_trigger = float(recent["high"].max())
    risk = max(confirmation_trigger - zone_low, atr_value or 0.01)
    rr_targets = {
        "1R": round(confirmation_trigger + risk, 4),
        "2R": round(confirmation_trigger + risk * 2, 4),
        "3R": round(confirmation_trigger + risk * 3, 4),
    }
    is_valid = price_in_zone and (rejection or fvg_fill) and dry_up
    explanation = (
        "Price has retested the breakout region with lower participation and a reclaim-style candle."
        if is_valid
        else "No lower-risk pullback entry is active right now."
    )
    return PullbackSignal(
        is_valid=is_valid,
        setup_type="retest_pullback",
        entry_zone=(round(zone_low, 4), round(zone_high, 4)),
        confirmation_trigger=round(confirmation_trigger, 4),
        stop_zone=(round(zone_low - atr_value * 0.35, 4), round(zone_low, 4)),
        rr_targets=rr_targets,
        explanation=explanation,
        metrics={
            "dry_up": dry_up,
            "rejection": rejection,
            "fvg_fill": fvg_fill,
            "price_in_zone": price_in_zone,
        },
    )
