from __future__ import annotations

from typing import Any

from signals.models import SetupSignal


def build_execution_plan(
    setup: SetupSignal,
    *,
    capital_base: float,
    risk_per_trade_pct: float,
    portfolio_exposure_pct: float = 0.0,
    correlation_penalty_pct: float = 0.0,
) -> dict[str, Any]:
    if setup.pullback and setup.pullback.is_valid and setup.pullback.confirmation_trigger:
        entry = float(setup.pullback.confirmation_trigger)
        stop = float(setup.pullback.stop_zone[0]) if setup.pullback.stop_zone else float(setup.breakout.invalidation_level or entry * 0.97)
    else:
        entry = float(setup.breakout.buffered_level or setup.breakout.current_price)
        stop = float(setup.breakout.invalidation_level or (entry * 0.96))

    atr_stop = min(stop, entry - max(abs(entry - stop), 0.01))
    structure_stop = float(setup.structure.invalidation_level or stop)
    risk_per_share = max(entry - min(atr_stop, structure_stop), 0.01)
    capital_risk = capital_base * (risk_per_trade_pct / 100.0)
    raw_position = capital_risk / risk_per_share
    adjusted_position = raw_position * max(0.1, 1.0 - (portfolio_exposure_pct / 100.0) - (correlation_penalty_pct / 100.0))

    target_1r = entry + risk_per_share
    target_2r = entry + risk_per_share * 2
    target_3r = entry + risk_per_share * 3

    warnings: list[str] = []
    if setup.event_risk_days is not None and setup.event_risk_days <= 7:
        warnings.append("Event risk is near. Reduce size or skip if the catalyst window matters.")
    if setup.volume and setup.volume.volume_ratio < 1.3:
        warnings.append("Participation is not ideal for a high-conviction breakout.")
    if setup.breakout.distance_pct is not None and setup.breakout.distance_pct > 6:
        warnings.append("Price is getting extended away from the breakout reference.")

    return {
        "entry": round(entry, 4),
        "stop": round(min(atr_stop, structure_stop), 4),
        "atr_stop": round(atr_stop, 4),
        "structure_stop": round(structure_stop, 4),
        "target_1r": round(target_1r, 4),
        "target_2r": round(target_2r, 4),
        "target_3r": round(target_3r, 4),
        "trail_reference": round(max(target_1r, float(setup.breakout.current_price)), 4),
        "position_size_shares": int(max(adjusted_position, 0)),
        "capital_risk": round(capital_risk, 2),
        "portfolio_exposure_pct": portfolio_exposure_pct,
        "correlation_penalty_pct": correlation_penalty_pct,
        "warnings": warnings,
    }
