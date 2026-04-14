from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signals.common import atr, ema


@dataclass(slots=True)
class BacktestResult:
    trades: pd.DataFrame
    summary: pd.DataFrame
    walk_forward: pd.DataFrame
    monte_carlo: pd.DataFrame


def _build_breakout_entries(frame: pd.DataFrame, lookback: int = 20, buffer_pct: float = 0.5, min_volume_ratio: float = 1.5) -> pd.Series:
    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    breakout_level = working["high"].shift(1).rolling(lookback, min_periods=lookback).max()
    avg_volume = working["volume"].shift(1).rolling(20, min_periods=20).mean()
    ema20 = ema(working["close"], 20)
    ema50 = ema(working["close"], 50)
    entries = (
        (working["close"] > breakout_level * (1 + buffer_pct / 100.0))
        & (working["volume"] > avg_volume * min_volume_ratio)
        & (working["close"] > ema20)
        & (working["close"] > ema50)
    )
    return entries.fillna(False)


def run_backtest(
    frame: pd.DataFrame,
    *,
    lookback: int = 20,
    buffer_pct: float = 0.5,
    min_volume_ratio: float = 1.5,
    slippage_bps: float = 5.0,
    brokerage_bps: float = 3.0,
    taxes_bps: float = 2.0,
    max_holding_bars: int = 20,
) -> BacktestResult:
    working = frame.copy().sort_values("datetime").reset_index(drop=True)
    working["atr14"] = atr(working, 14)
    entries = _build_breakout_entries(working, lookback, buffer_pct, min_volume_ratio)

    trades: list[dict[str, float | str]] = []
    idx = 0
    while idx < len(working) - 2:
        if not bool(entries.iloc[idx]):
            idx += 1
            continue
        entry_idx = idx + 1
        if entry_idx >= len(working):
            break
        entry_price = float(working.iloc[entry_idx]["open"]) * (1 + slippage_bps / 10000.0)
        atr_value = float(working.iloc[idx]["atr14"]) if pd.notna(working.iloc[idx]["atr14"]) else entry_price * 0.03
        stop = entry_price - atr_value * 1.5
        target = entry_price + (entry_price - stop) * 2.0
        exit_price = float(working.iloc[min(len(working) - 1, entry_idx + max_holding_bars)]["close"])
        exit_reason = "time_stop"
        exit_idx = min(len(working) - 1, entry_idx + max_holding_bars)
        for probe in range(entry_idx, min(len(working), entry_idx + max_holding_bars + 1)):
            candle = working.iloc[probe]
            if float(candle["low"]) <= stop:
                exit_price = stop * (1 - slippage_bps / 10000.0)
                exit_reason = "stop"
                exit_idx = probe
                break
            if float(candle["high"]) >= target:
                exit_price = target * (1 - slippage_bps / 10000.0)
                exit_reason = "target_2r"
                exit_idx = probe
                break
        gross_return = (exit_price / entry_price) - 1.0
        total_cost_bps = slippage_bps * 2 + brokerage_bps + taxes_bps
        net_return = gross_return - (total_cost_bps / 10000.0)
        r_multiple = (exit_price - entry_price) / max(entry_price - stop, 1e-9)
        trades.append(
            {
                "entry_time": working.iloc[entry_idx]["datetime"],
                "exit_time": working.iloc[exit_idx]["datetime"],
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "stop": round(stop, 4),
                "target": round(target, 4),
                "return_pct": round(net_return * 100.0, 4),
                "r_multiple": round(r_multiple, 4),
                "exit_reason": exit_reason,
                "bars_held": exit_idx - entry_idx,
            },
        )
        idx = exit_idx + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        summary = pd.DataFrame([{"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}])
        walk_forward = pd.DataFrame(
            [
                {"segment": "in_sample", "trades": 0, "win_rate": 0.0, "expectancy_r": 0.0},
                {"segment": "out_of_sample", "trades": 0, "win_rate": 0.0, "expectancy_r": 0.0},
            ],
        )
        return BacktestResult(trades=trades_df, summary=summary, walk_forward=walk_forward, monte_carlo=summary.copy())

    equity = (1 + trades_df["return_pct"] / 100.0).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    summary = pd.DataFrame(
        [
            {
                "trades": int(len(trades_df)),
                "win_rate": round(float((trades_df["return_pct"] > 0).mean() * 100.0), 2),
                "expectancy_r": round(float(trades_df["r_multiple"].mean()), 4),
                "profit_factor": round(
                    float(trades_df.loc[trades_df["return_pct"] > 0, "return_pct"].sum() / abs(trades_df.loc[trades_df["return_pct"] <= 0, "return_pct"].sum()))
                    if abs(trades_df.loc[trades_df["return_pct"] <= 0, "return_pct"].sum()) > 0
                    else 0.0,
                    4,
                ),
                "max_drawdown_pct": round(float(drawdown.min() * 100.0), 2),
                "cagr_like_pct": round(float((equity.iloc[-1] ** (252 / max(len(working), 1)) - 1.0) * 100.0), 2),
                "sharpe_like": round(float((trades_df["return_pct"].mean() / max(trades_df["return_pct"].std(ddof=0), 1e-9)) * np.sqrt(12)), 4),
                "sortino_like": round(
                    float(
                        trades_df["return_pct"].mean()
                        / max(trades_df.loc[trades_df["return_pct"] < 0, "return_pct"].std(ddof=0), 1e-9)
                        * np.sqrt(12)
                    ),
                    4,
                ),
            },
        ],
    )

    split = int(len(working) * 0.7)
    walk_rows = []
    for label, segment in [("in_sample", working.iloc[:split].copy()), ("out_of_sample", working.iloc[split:].copy())]:
        if len(segment) <= 50:
            walk_rows.append({"segment": label, "trades": 0, "win_rate": 0.0, "expectancy_r": 0.0})
            continue
        segment_entries = _build_breakout_entries(segment, lookback, buffer_pct, min_volume_ratio)
        walk_rows.append(
            {
                "segment": label,
                "trades": int(segment_entries.sum()),
                "win_rate": float((segment_entries.mean() * 100.0)),
                "expectancy_r": float(segment_entries.mean() * 2.0 - (1 - segment_entries.mean())),
            },
        )
    walk_forward = pd.DataFrame(walk_rows)

    rng = np.random.default_rng(42)
    trade_r = trades_df["r_multiple"].to_numpy(dtype=float)
    runs = []
    for run_idx in range(200):
        sampled = rng.choice(trade_r, size=len(trade_r), replace=True)
        runs.append({"run": run_idx, "ending_r": float(sampled.sum()), "worst_r": float(sampled.min()) if len(sampled) else 0.0})
    monte_carlo = pd.DataFrame(runs)
    return BacktestResult(trades=trades_df, summary=summary, walk_forward=walk_forward, monte_carlo=monte_carlo)
