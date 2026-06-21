/**
 * Momentum confirmation (RSI + MACD). STOCKER accuracy addition — STOPICK
 * computed RSI but never used it. A healthy-but-not-overbought RSI plus a
 * rising MACD histogram is treated as confirmation; overbought RSI is penalized.
 */
import type { Bar, MomentumSignal } from "./types";
import { bounded, closes, last, macd, round, rsi } from "./indicators";

export function analyzeMomentum(bars: Bar[]): MomentumSignal {
  const c = closes(bars);
  const rsiSeries = rsi(c, 14);
  const rsiVal = last(rsiSeries, 50);
  const { macd: macdLine, signal, histogram } = macd(c);
  const macdVal = last(macdLine, 0);
  const signalVal = last(signal, 0);
  const histVal = last(histogram, 0);
  const prevHist = histogram.length >= 2 ? histogram[histogram.length - 2] : 0;

  let rsiState: string;
  if (rsiVal < 35) rsiState = "oversold";
  else if (rsiVal <= 60) rsiState = "healthy";
  else if (rsiVal <= 75) rsiState = "strong";
  else rsiState = "overbought";

  // RSI quality: peak around 55–68 (trend momentum), decays into overbought.
  let rsiQuality: number;
  if (rsiVal <= 50) rsiQuality = bounded((rsiVal - 35) / 15) * 0.6;
  else if (rsiVal <= 68) rsiQuality = 0.6 + bounded((rsiVal - 50) / 18) * 0.4;
  else rsiQuality = bounded(1 - (rsiVal - 68) / 20);

  const macdBullish = histVal > 0 || (histVal > prevHist && macdVal > signalVal);
  const macdQuality = bounded(0.5 + Math.sign(histVal) * 0.3 + (histVal > prevHist ? 0.2 : -0.1));

  const score = round(bounded(rsiQuality * 0.55 + macdQuality * 0.45) * 100, 2);
  const explanation =
    `RSI ${round(rsiVal, 1)} (${rsiState}); MACD histogram ${round(histVal, 4)} ` +
    `${macdBullish ? "supports" : "does not support"} continuation.`;

  return {
    rsi: round(rsiVal, 2),
    rsiState,
    macd: round(macdVal, 4),
    macdSignal: round(signalVal, 4),
    macdHistogram: round(histVal, 4),
    macdBullish,
    score,
    explanation,
  };
}
