import { BacktestClient } from "@/components/backtest/BacktestClient";

export const metadata = {
  title: "Backtest — STOCKER",
  description: "Walk-forward breakout backtest with Monte Carlo trade-sequence resampling.",
};

export default function BacktestPage() {
  return <BacktestClient />;
}
