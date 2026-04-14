from __future__ import annotations

from config import AppConfig
from signals.models import SetupSignal


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def score_setup_signal(setup: SetupSignal, config: AppConfig, profile_name: str) -> tuple[float, str, dict[str, float]]:
    profile = config.scoring_profiles[profile_name]
    breakout = setup.breakout
    pullback = setup.pullback
    volume = setup.volume
    relative_strength = setup.relative_strength
    regime = setup.regime
    trend = setup.trend
    structure = setup.structure

    rr_estimate = 0.0
    if setup.execution_plan:
        rr_estimate = float(setup.execution_plan.get("target_2r", 0.0) - setup.execution_plan.get("entry", 0.0)) / max(
            abs(setup.execution_plan.get("entry", 0.0) - setup.execution_plan.get("stop", 0.0)),
            1e-9,
        )

    event_score = 1.0 if setup.event_risk_days in {None} or setup.event_risk_days > 7 else 0.35
    pullback_component = 0.5 if pullback is None else 1.0 if pullback.is_valid else 0.2
    overhead = breakout.overhead_resistance_pct
    headroom_component = 1.0 if overhead is None else 0.75 if overhead > 8 else 0.45 if overhead > 4 else 0.1
    component_values = {
        "trend_alignment": _bounded((trend.strength_score / 100.0 + trend.alignment_confidence / 100.0) / 2),
        "structure_quality": _bounded(
            (float(structure.bos) * 0.45)
            + (float(not structure.fvg_mitigated and structure.fvg_zone is not None) * 0.25)
            + (float(structure.liquidity_sweep) * 0.15)
            + (float(not structure.choch) * 0.15),
        ),
        "breakout_quality": _bounded(
            (float(breakout.is_valid) * 0.4)
            + ((breakout.candle_quality / 100.0) * 0.25)
            + ((breakout.tightness_score / 100.0) * 0.2)
            + (_bounded((breakout.volume_expansion - 1.0) / 1.5) * 0.15),
        ),
        "pullback_quality": _bounded(pullback_component),
        "volume_confirmation": _bounded(
            (
                ((volume.volume_ratio - 1.0) / 1.5 if volume else 0.0) * 0.55
                + (float(volume.obv_confirmation) * 0.2 if volume else 0.0)
                + (float(volume.vwap_alignment) * 0.15 if volume else 0.0)
                + (float(volume.dry_up_before_expansion) * 0.1 if volume else 0.0)
            ),
        ),
        "volatility_regime": 0.8 if regime and regime.volatility_state in {"compressed", "normal"} else 0.45,
        "relative_strength": _bounded((relative_strength.score / 100.0) if relative_strength else 0.0),
        "htf_headroom": headroom_component,
        "rr_ratio": _bounded(rr_estimate / 3.0),
        "market_regime": 1.0 if regime and regime.direction == setup.direction else 0.55 if regime and regime.direction == "neutral" else 0.15,
        "event_risk": event_score,
        "index_alignment": 1.0 if regime and regime.direction == trend.direction else 0.5,
    }

    total_weight = sum(profile.weights.values())
    weighted_score = sum(component_values[key] * weight for key, weight in profile.weights.items())
    final_score = round((weighted_score / total_weight) * 100.0, 2) if total_weight else 0.0

    grade = "Reject"
    for label, threshold in sorted(profile.grade_thresholds.items(), key=lambda item: item[1], reverse=True):
        if final_score >= threshold:
            grade = label
            break
    breakdown = {key: round(value * 100.0, 2) for key, value in component_values.items()}
    return final_score, grade, breakdown
