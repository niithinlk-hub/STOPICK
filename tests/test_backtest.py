from __future__ import annotations

from backtest.engine import run_backtest


def test_run_backtest_returns_summary(bullish_df) -> None:
    result = run_backtest(bullish_df)
    assert "trades" in result.summary.columns
    assert "win_rate" in result.summary.columns
    assert result.walk_forward.shape[0] == 2
