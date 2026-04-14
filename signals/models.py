from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TrendSignal(BaseModel):
    direction: str
    strength_score: float
    alignment_confidence: float
    timeframe_scores: dict[str, float]
    metrics: dict[str, float | str | None]


class StructureSignal(BaseModel):
    ticker: str
    market: str
    timeframe: str
    direction: str
    structure_type: str
    key_levels: dict[str, float | None]
    retest_zone: tuple[float, float] | None = None
    invalidation_level: float | None = None
    bos: bool = False
    choch: bool = False
    liquidity_sweep: bool = False
    equal_highs: bool = False
    equal_lows: bool = False
    inducement: bool = False
    order_block_zone: tuple[float, float] | None = None
    fvg_zone: tuple[float, float] | None = None
    fvg_mitigated: bool = False
    explanation: str = ""


class BreakoutSignal(BaseModel):
    is_valid: bool
    pattern_name: str
    direction: str
    breakout_level: float | None
    buffered_level: float | None
    current_price: float
    distance_pct: float | None
    candle_quality: float
    tightness_score: float
    volume_expansion: float
    overhead_resistance_pct: float | None
    invalidation_level: float | None
    explanation: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class PullbackSignal(BaseModel):
    is_valid: bool
    setup_type: str
    entry_zone: tuple[float, float] | None
    confirmation_trigger: float | None
    stop_zone: tuple[float, float] | None
    rr_targets: dict[str, float]
    explanation: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class VolumeSignal(BaseModel):
    volume_ratio: float
    relative_volume: float
    obv_confirmation: bool
    vwap_alignment: bool
    anchored_vwap: float | None
    dry_up_before_expansion: bool
    penalty_flags: list[str]


class RelativeStrengthSignal(BaseModel):
    benchmark_symbol: str
    sector_benchmark_symbol: str | None = None
    score: float
    trend_persistence: float
    smoothness: float
    one_week_alpha: float
    one_month_alpha: float
    three_month_alpha: float
    explanation: str


class RegimeSignal(BaseModel):
    market: str
    benchmark_symbol: str
    direction: str
    trend_strength: float
    volatility_state: str
    breadth_like_proxy: float
    explanation: str


class SetupSignal(BaseModel):
    ticker: str
    market: str
    exchange: str
    country: str
    sector: str
    timeframe: str
    setup_family: str
    direction: str
    trend: TrendSignal
    structure: StructureSignal
    breakout: BreakoutSignal
    pullback: PullbackSignal | None = None
    volume: VolumeSignal | None = None
    relative_strength: RelativeStrengthSignal | None = None
    regime: RegimeSignal | None = None
    score: float = 0.0
    grade: str = "Reject"
    reasons_for: list[str] = Field(default_factory=list)
    reasons_against: list[str] = Field(default_factory=list)
    execution_plan: dict[str, Any] = Field(default_factory=dict)
    risk_warnings: list[str] = Field(default_factory=list)
    event_risk_days: int | None = None
