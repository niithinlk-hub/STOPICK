from __future__ import annotations

from pathlib import Path

from config import load_app_config
from risk.planner import build_execution_plan
from scoring.engine import score_setup_signal
from signals.breakout import find_best_breakout
from signals.context import analyze_market_regime, analyze_relative_strength, analyze_volume_participation
from signals.models import SetupSignal
from signals.pullback import find_pullback_entry
from signals.structure import analyze_market_structure
from signals.trend_alignment import analyze_trend_alignment


def test_score_setup_signal_returns_grade(bullish_df, benchmark_df) -> None:
    config = load_app_config(Path(__file__).resolve().parent.parent)
    trend = analyze_trend_alignment({"1d": bullish_df, "4h": bullish_df, "1h": bullish_df, "15m": bullish_df})
    structure = analyze_market_structure(bullish_df, ticker="TEST", market="US", timeframe="1d")
    relative_strength = analyze_relative_strength(bullish_df, benchmark_df, benchmark_symbol="SPY")
    breakout = find_best_breakout(
        bullish_df,
        market="US",
        trend_signal=trend,
        structure_signal=structure,
        relative_strength_score=relative_strength.score,
    )
    pullback = find_pullback_entry(bullish_df, breakout, structure)
    volume = analyze_volume_participation(bullish_df, breakout.breakout_level)
    regime = analyze_market_regime(benchmark_df, market="US", benchmark_symbol="SPY")

    setup = SetupSignal(
        ticker="TEST",
        market="US",
        exchange="NASDAQ",
        country="US",
        sector="Technology",
        timeframe="1d",
        setup_family="breakout",
        direction="bullish",
        trend=trend,
        structure=structure,
        breakout=breakout,
        pullback=pullback,
        volume=volume,
        relative_strength=relative_strength,
        regime=regime,
        reasons_for=["trend", "structure"],
        reasons_against=[],
    )
    setup.execution_plan = build_execution_plan(setup, capital_base=config.runtime.capital_base, risk_per_trade_pct=config.runtime.risk_per_trade_pct)
    score, grade, breakdown = score_setup_signal(setup, config, "bullish_breakout")
    assert score >= 0
    assert grade in {"A+", "A", "B", "C", "Reject"}
    assert "trend_alignment" in breakdown
