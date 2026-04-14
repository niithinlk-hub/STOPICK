from __future__ import annotations

from signals.breakout import find_best_breakout
from signals.context import analyze_market_regime, analyze_relative_strength, analyze_volume_participation
from signals.pullback import find_pullback_entry
from signals.structure import analyze_market_structure
from signals.trend_alignment import analyze_trend_alignment


def test_signal_stack_returns_bullish_context(bullish_df, benchmark_df) -> None:
    frame_map = {"1d": bullish_df, "4h": bullish_df.tail(260), "1h": bullish_df.tail(240), "15m": bullish_df.tail(220)}
    trend = analyze_trend_alignment(frame_map)
    structure = analyze_market_structure(bullish_df, ticker="TEST", market="US", timeframe="1d")
    relative_strength = analyze_relative_strength(bullish_df, benchmark_df, benchmark_symbol="SPY")
    breakout = find_best_breakout(
        bullish_df,
        market="US",
        trend_signal=trend,
        structure_signal=structure,
        relative_strength_score=relative_strength.score,
    )
    volume = analyze_volume_participation(bullish_df, breakout.breakout_level)
    pullback = find_pullback_entry(bullish_df, breakout, structure)
    regime = analyze_market_regime(benchmark_df, market="US", benchmark_symbol="SPY")

    assert trend.direction in {"bullish", "neutral"}
    assert structure.structure_type in {"BOS", "range", "CHOCH"}
    assert breakout.pattern_name != "None"
    assert volume.volume_ratio >= 1.0
    assert regime.benchmark_symbol == "SPY"
    assert pullback.setup_type == "retest_pullback"
