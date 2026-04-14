from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import AppConfig
from data.loaders import DataEngine
from data.symbols import SymbolRecord, benchmark_for_market, display_symbol, load_watchlist_file, parse_manual_watchlist
from risk.planner import build_execution_plan
from scoring.engine import score_setup_signal
from signals.breakout import find_best_breakout
from signals.common import atr, rolling_percentile
from signals.context import analyze_market_regime, analyze_relative_strength, analyze_volume_participation
from signals.models import SetupSignal
from signals.pullback import find_pullback_entry
from signals.structure import analyze_market_structure
from signals.trend_alignment import analyze_trend_alignment


@dataclass(slots=True)
class ScanBundle:
    setups: list[SetupSignal]
    results: pd.DataFrame
    frame_cache: dict[str, dict[str, pd.DataFrame]]
    failures: dict[str, str]
    benchmark_frames: dict[str, pd.DataFrame]
    scanned_symbols: int
    successful_symbols: int
    notes: list[str]


def _timeframes_for_scan(scan_timeframe: str) -> list[str]:
    if scan_timeframe == "1d":
        return ["1d"]
    if scan_timeframe == "4h":
        return ["1d", "4h"]
    if scan_timeframe == "1h":
        return ["1d", "4h", "1h"]
    return ["1d", "4h", "1h", "15m"]


def _records_from_source(
    config: AppConfig,
    *,
    country: str,
    source: str,
    manual_watchlist: str = "",
    uploaded_watchlist_text: str | None = None,
    uploaded_watchlist_frame: pd.DataFrame | None = None,
) -> list[SymbolRecord]:
    records: list[SymbolRecord] = []
    if source == "sample":
        if country in {"NSE", "BOTH"}:
            records.extend(SymbolRecord(symbol=symbol, market="NSE", exchange="NSE") for symbol in load_watchlist_file(config.universe_files["nse_sample"], "NSE"))
        if country in {"US", "BOTH"}:
            records.extend(SymbolRecord(symbol=symbol, market="US", exchange="US") for symbol in load_watchlist_file(config.universe_files["us_sample"], "US"))
    else:
        if uploaded_watchlist_frame is not None and not uploaded_watchlist_frame.empty:
            symbol_column = "symbol" if "symbol" in uploaded_watchlist_frame.columns else "ticker"
            for _, row in uploaded_watchlist_frame.iterrows():
                market = str(row.get("market", country if country != "BOTH" else "US")).upper()
                if country != "BOTH" and market != country:
                    continue
                symbol = str(row.get(symbol_column, "")).strip()
                if not symbol:
                    continue
                records.append(
                    SymbolRecord(
                        symbol=symbol if market == "US" or symbol.endswith(".NS") else f"{symbol}.NS",
                        market=market,
                        exchange=str(row.get("exchange", market)),
                        sector=str(row.get("sector", "Unknown")),
                        market_cap_bucket=str(row.get("market_cap_bucket", "Unknown")),
                    ),
                )
        text = uploaded_watchlist_text or manual_watchlist
        for market in (["NSE", "US"] if country == "BOTH" else [country]):
            records.extend(SymbolRecord(symbol=symbol, market=market, exchange=market) for symbol in parse_manual_watchlist(text, market))
    unique: dict[str, SymbolRecord] = {record.symbol: record for record in records}
    return list(unique.values())[: config.runtime.max_symbols_per_scan]


def _event_map(data_engine: DataEngine, records: list[SymbolRecord]) -> tuple[dict[str, int | None], list[str]]:
    default_map = {display_symbol(record.symbol): None for record in records}
    notes: list[str] = []
    if not records:
        return default_map, notes
    if len(records) > 10:
        notes.append("Event-calendar lookup was skipped for this larger scan to keep the scanner responsive.")
        return default_map, notes
    try:
        calendar = data_engine.provider.fetch_corporate_calendar(record.symbol for record in records)
    except Exception as exc:
        notes.append(f"Event-calendar lookup was unavailable: {exc}")
        return default_map, notes
    if calendar.empty:
        return default_map, notes
    calendar["event_date"] = pd.to_datetime(calendar["event_date"], utc=True, errors="coerce")
    now = pd.Timestamp.utcnow()
    event_days: dict[str, int | None] = {}
    for _, row in calendar.iterrows():
        if pd.isna(row["event_date"]):
            event_days[str(row["symbol"])] = None
        else:
            event_days[str(row["symbol"])] = int((row["event_date"] - now).days)
    for record in records:
        event_days.setdefault(display_symbol(record.symbol), None)
    return event_days, notes


def scan_market(
    config: AppConfig,
    *,
    country: str,
    source: str,
    scan_timeframe: str,
    minimum_score: float,
    setup_mode: str,
    manual_watchlist: str = "",
    uploaded_watchlist_text: str | None = None,
    uploaded_watchlist_frame: pd.DataFrame | None = None,
    refresh_data: bool = False,
) -> ScanBundle:
    data_engine = DataEngine(config)
    records = _records_from_source(
        config,
        country=country,
        source=source,
        manual_watchlist=manual_watchlist,
        uploaded_watchlist_text=uploaded_watchlist_text,
        uploaded_watchlist_frame=uploaded_watchlist_frame,
    )
    event_days, notes = _event_map(data_engine, records)

    frame_cache: dict[str, dict[str, pd.DataFrame]] = {}
    benchmark_frames: dict[str, pd.DataFrame] = {}
    failures: dict[str, str] = {}
    setups: list[SetupSignal] = []
    setup_metrics: dict[str, dict[str, float]] = {}

    multi_timeframes = _timeframes_for_scan(scan_timeframe)
    for record in records:
        frame_map: dict[str, pd.DataFrame] = {}
        try:
            for timeframe in multi_timeframes:
                if timeframe == "15m" and scan_timeframe not in {"15m", "5m"}:
                    continue
                frame_map[timeframe] = data_engine.fetch_symbol(record, timeframe, refresh=refresh_data)
            scan_frame = frame_map.get(scan_timeframe)
            if scan_frame is None or scan_frame.empty:
                scan_frame = frame_map.get("1d")
            if scan_frame is None or scan_frame.empty:
                failures[record.symbol] = "No scan frame data."
                continue

            benchmark_symbol = benchmark_for_market(record.market, config.data.benchmark_map)
            if benchmark_symbol not in benchmark_frames:
                benchmark_record = SymbolRecord(symbol=benchmark_symbol, market=record.market, exchange=record.exchange)
                benchmark_frames[benchmark_symbol] = data_engine.fetch_symbol(benchmark_record, "1d", refresh=refresh_data)

            trend = analyze_trend_alignment(frame_map, adx_threshold=20.0)
            structure = analyze_market_structure(scan_frame, ticker=display_symbol(record.symbol), market=record.market, timeframe=scan_timeframe)
            volume = analyze_volume_participation(scan_frame, structure.key_levels.get("bos_level"))
            relative_strength = analyze_relative_strength(scan_frame, benchmark_frames[benchmark_symbol], benchmark_symbol=benchmark_symbol)
            regime = analyze_market_regime(benchmark_frames[benchmark_symbol], market=record.market, benchmark_symbol=benchmark_symbol)
            breakout = find_best_breakout(
                scan_frame,
                market=record.market,
                trend_signal=trend,
                structure_signal=structure,
                relative_strength_score=relative_strength.score,
                breakout_buffer_pct=0.5,
                lookback=40,
                event_proximity_days=event_days.get(display_symbol(record.symbol)),
            )
            pullback = find_pullback_entry(scan_frame, breakout, structure)
            avg_volume_20 = round(float(scan_frame["volume"].tail(20).mean()), 2)
            atr_pct = round(float(rolling_percentile(atr(scan_frame, 14), 252).iloc[-1] * 100.0), 2) if len(scan_frame) >= 30 else 0.0

            for family in (["breakout", "pullback"] if setup_mode == "both" else [setup_mode]):
                if family == "breakout" and not breakout.is_valid:
                    continue
                if family == "pullback" and not pullback.is_valid:
                    continue
                setup = SetupSignal(
                    ticker=display_symbol(record.symbol),
                    market=record.market,
                    exchange=record.exchange,
                    country=record.market,
                    sector=record.sector,
                    timeframe=scan_timeframe,
                    setup_family=family,
                    direction=trend.direction if trend.direction != "neutral" else "bullish",
                    trend=trend,
                    structure=structure,
                    breakout=breakout,
                    pullback=pullback if family == "pullback" else None,
                    volume=volume,
                    relative_strength=relative_strength,
                    regime=regime,
                    reasons_for=[
                        breakout.explanation,
                        structure.explanation,
                        relative_strength.explanation,
                        "Volume participation confirms the move." if volume.volume_ratio >= 1.5 else "Participation is acceptable but not exceptional.",
                    ],
                    reasons_against=list(volume.penalty_flags),
                    event_risk_days=event_days.get(display_symbol(record.symbol)),
                )
                setup.execution_plan = build_execution_plan(
                    setup,
                    capital_base=config.runtime.capital_base,
                    risk_per_trade_pct=config.runtime.risk_per_trade_pct,
                )
                setup.risk_warnings = list(setup.execution_plan.get("warnings", []))
                profile_name = "bullish_pullback" if family == "pullback" else "bullish_breakout"
                score, grade, _ = score_setup_signal(setup, config, profile_name)
                setup.score = score
                setup.grade = grade
                if score >= minimum_score:
                    setups.append(setup)
                    setup_metrics[f"{setup.ticker}|{family}"] = {"avg_volume_20": avg_volume_20, "atr_percentile": atr_pct}
            frame_cache[record.symbol] = frame_map
        except Exception as exc:
            failures[record.symbol] = str(exc)

    rows = []
    for setup in setups:
        metrics = setup_metrics.get(f"{setup.ticker}|{setup.setup_family}", {})
        rows.append(
            {
                "ticker": setup.ticker,
                "market": setup.market,
                "exchange": setup.exchange,
                "sector": setup.sector,
                "market_cap_bucket": next((record.market_cap_bucket for record in records if display_symbol(record.symbol) == setup.ticker), "Unknown"),
                "timeframe": setup.timeframe,
                "setup_family": setup.setup_family,
                "pattern": setup.breakout.pattern_name,
                "score": setup.score,
                "grade": setup.grade,
                "trend_direction": setup.trend.direction,
                "trend_strength": setup.trend.strength_score,
                "breakout_level": setup.breakout.breakout_level,
                "current_price": setup.breakout.current_price,
                "distance_pct": setup.breakout.distance_pct,
                "avg_volume_20": metrics.get("avg_volume_20"),
                "volume_ratio": setup.volume.volume_ratio if setup.volume else None,
                "atr_percentile": metrics.get("atr_percentile"),
                "relative_strength_score": setup.relative_strength.score if setup.relative_strength else None,
                "event_risk_days": setup.event_risk_days,
                "why_this_qualified": " | ".join(setup.reasons_for[:3]),
                "execution_entry": setup.execution_plan.get("entry"),
                "execution_stop": setup.execution_plan.get("stop"),
                "target_2r": setup.execution_plan.get("target_2r"),
            },
        )
    results = pd.DataFrame(rows).sort_values(["score", "relative_strength_score"], ascending=[False, False], kind="mergesort") if rows else pd.DataFrame()
    return ScanBundle(
        setups=setups,
        results=results,
        frame_cache=frame_cache,
        failures=failures,
        benchmark_frames=benchmark_frames,
        scanned_symbols=len(records),
        successful_symbols=len(frame_cache),
        notes=notes,
    )
