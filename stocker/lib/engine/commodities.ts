/**
 * Commodity multi-horizon predictor. For each commodity we pull daily OHLCV and
 * derive a directional read + confidence for four horizons (1D / 1W / 1M / 2M)
 * from a transparent blend of trend, moving-average alignment, slope, momentum
 * and oscillator signals — then optionally fuse a news-sentiment score.
 *
 * Everything here is pure + deterministic; the route supplies bars + news score.
 */
import type { Bar } from "./types";
import { adx, atr, closes, ema, efficiencyRatio, highs, last, linregSlope, lows, macd, round, rsi, sma } from "./indicators";

export interface CommodityDef {
  key: string;
  name: string;
  symbol: string; // Yahoo symbol
  unit: string;
  newsQuery: string;
}

export const COMMODITIES: CommodityDef[] = [
  { key: "gold", name: "Gold", symbol: "GC=F", unit: "oz", newsQuery: "gold price" },
  { key: "silver", name: "Silver", symbol: "SI=F", unit: "oz", newsQuery: "silver price" },
  { key: "crude", name: "Crude Oil (WTI)", symbol: "CL=F", unit: "bbl", newsQuery: "crude oil price WTI" },
];

export type HorizonKey = "1D" | "1W" | "1M" | "2M";

interface HorizonCfg {
  key: HorizonKey;
  label: string;
  fast: number;
  slow: number;
  slopeWin: number;
  roc: number;
  days: number; // approx trading days in the horizon
  newsW: number; // weight given to news at this horizon
}

export const HORIZONS: HorizonCfg[] = [
  { key: "1D", label: "1 Day", fast: 5, slow: 10, slopeWin: 5, roc: 3, days: 1, newsW: 0.45 },
  { key: "1W", label: "1 Week", fast: 10, slow: 20, slopeWin: 10, roc: 5, days: 5, newsW: 0.4 },
  { key: "1M", label: "1 Month", fast: 20, slow: 50, slopeWin: 20, roc: 21, days: 21, newsW: 0.28 },
  { key: "2M", label: "2 Months", fast: 50, slow: 100, slopeWin: 50, roc: 42, days: 42, newsW: 0.18 },
];

export interface HorizonReadings {
  price: number;
  rsi: number;
  adx: number;
  trendSlopePctPerWeek: number;
  atrPct: number;
  emaFast: number;
  emaSlow: number;
}

export interface HorizonPrediction {
  horizon: HorizonKey;
  label: string;
  direction: "bullish" | "bearish" | "neutral";
  action: string;
  confidence: number; // 0–100
  pattern: string;
  technicalScore: number; // 0–100, technical conviction alone
  newsScore: number | null; // -1..1 if news fused
  agreement: number; // -1 / 0 / +1 news vs technical
  readings: HorizonReadings;
}

export interface NewsSentiment {
  score: number; // -1..1
  label: string;
  summary: string;
  drivers: string[];
  headlines: { title: string; source: string }[];
}

export interface CommodityNewsBlock {
  score: number;
  label: string;
  summary: string;
  drivers: string[];
  headlines: { title: string; source: string }[];
  enabled: boolean;
  error?: string;
}

export interface CommodityPrediction {
  key: string;
  name: string;
  symbol: string;
  unit: string;
  price: number;
  dayChangePct: number | null;
  horizons: HorizonPrediction[];
  news: CommodityNewsBlock | null;
  asOf: number;
}

export interface CommoditiesResponse {
  commodities: CommodityPrediction[];
  generatedAt: number;
  newsConfigured: boolean;
  notes: string[];
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const sign = (v: number) => (v > 0 ? 1 : v < 0 ? -1 : 0);

interface TechRead {
  net: number;
  conv: number;
  score: number;
  pattern: string;
  readings: HorizonReadings;
}

function horizonTechnical(bars: Bar[], cfg: HorizonCfg): TechRead {
  const c = closes(bars);
  const n = c.length;
  const price = last(c);
  const atrPct = (last(atr(bars, 14)) || price * 0.01) / price;
  const horizonATR = Math.max(atrPct * Math.sqrt(cfg.days), 0.002); // expected move over the horizon

  const emaFast = last(ema(c, cfg.fast));
  const emaSlow = last(ema(c, cfg.slow));
  const rsiV = last(rsi(c, 14));
  const adxV = last(adx(bars, 14)) || 0;
  const hist = last(macd(c).histogram);
  const slope = linregSlope(c, cfg.slopeWin); // price per bar
  const roc = n > cfg.roc ? c[n - 1] / c[n - 1 - cfg.roc] - 1 : 0;

  const trendVsSlow = clamp((price - emaSlow) / emaSlow / horizonATR, -1, 1);
  const maAlign = clamp((emaFast - emaSlow) / emaSlow / (horizonATR * 0.6), -1, 1);
  const slopeSig = clamp(((slope * cfg.slopeWin) / price) / horizonATR, -1, 1);
  const rocSig = clamp(roc / horizonATR, -1, 1);
  const rsiSig = clamp((rsiV - 50) / 30, -1, 1);
  const macdSig = clamp(hist / price / (atrPct * 0.5), -1, 1);

  const net = clamp(
    trendVsSlow * 0.28 + maAlign * 0.18 + slopeSig * 0.18 + rocSig * 0.16 + rsiSig * 0.1 + macdSig * 0.1,
    -1,
    1,
  );
  const conv = clamp(adxV / 40, 0, 1) * 0.5 + efficiencyRatio(c, cfg.slopeWin) * 0.5;
  const score = clamp(round(Math.abs(net) * 70 + conv * 30, 0), 0, 100);

  const readings: HorizonReadings = {
    price: round(price, price > 100 ? 2 : 3),
    rsi: round(rsiV, 1),
    adx: round(adxV, 1),
    trendSlopePctPerWeek: round(((slope * 5) / price) * 100, 2),
    atrPct: round(atrPct * 100, 2),
    emaFast: round(emaFast, price > 100 ? 2 : 3),
    emaSlow: round(emaSlow, price > 100 ? 2 : 3),
  };

  return { net, conv, score, pattern: pickPattern(bars, cfg, net, rsiV, adxV), readings };
}

function pickPattern(bars: Bar[], cfg: HorizonCfg, net: number, rsiV: number, adxV: number): string {
  const c = closes(bars);
  const h = highs(bars);
  const l = lows(bars);
  const price = last(c);
  const lookback = Math.min(Math.max(cfg.slow, 20), c.length - 1);
  const recentHigh = Math.max(...h.slice(-lookback));
  const recentLow = Math.min(...l.slice(-lookback));

  const smaFast = sma(c, cfg.fast);
  const smaSlow = sma(c, cfg.slow);
  const fNow = last(smaFast);
  const sNow = last(smaSlow);
  const fPrev = smaFast[smaFast.length - 2];
  const sPrev = smaSlow[smaSlow.length - 2];
  const crossedUp = Number.isFinite(fPrev) && Number.isFinite(sPrev) && fPrev <= sPrev && fNow > sNow;
  const crossedDown = Number.isFinite(fPrev) && Number.isFinite(sPrev) && fPrev >= sPrev && fNow < sNow;

  if (crossedUp) return `Bullish ${cfg.fast}/${cfg.slow} MA crossover`;
  if (crossedDown) return `Bearish ${cfg.fast}/${cfg.slow} MA crossover`;
  if (price >= recentHigh * 0.999 && net > 0) return `Breakout — new ${lookback}-day high`;
  if (price <= recentLow * 1.001 && net < 0) return `Breakdown — new ${lookback}-day low`;
  if (adxV < 18) return "Range / consolidation (low trend strength)";
  if (net > 0.12 && rsiV < 45) return "Uptrend pullback (buy-the-dip zone)";
  if (net > 0.12 && rsiV > 70) return "Overbought uptrend (extended)";
  if (net > 0.12) return "Established uptrend";
  if (net < -0.12 && rsiV > 55) return "Downtrend bounce (sell-the-rip zone)";
  if (net < -0.12 && rsiV < 30) return "Oversold downtrend (extended)";
  if (net < -0.12) return "Established downtrend";
  return "Neutral / mixed signals";
}

function actionFor(direction: "bullish" | "bearish" | "neutral", confidence: number): string {
  if (direction === "bullish") return confidence >= 65 ? "Buy" : "Accumulate on dips";
  if (direction === "bearish") return confidence >= 65 ? "Sell / avoid" : "Reduce / stay cautious";
  return "Hold — no clear edge";
}

/** Build the four-horizon prediction for one commodity. `news` may be null (technical-only). */
export function predictCommodity(bars: Bar[], news: NewsSentiment | null): HorizonPrediction[] {
  return HORIZONS.map((cfg) => {
    const tech = horizonTechnical(bars, cfg);
    const w = news ? cfg.newsW : 0;
    const finalNet = clamp(tech.net * (1 - w) + (news ? news.score : 0) * w, -1, 1);
    const agreement = news ? (news.score === 0 ? 0 : sign(tech.net) === sign(news.score) ? 1 : -1) : 0;

    let conf = Math.abs(finalNet) * 100;
    conf = conf * 0.7 + conf * 0.3 * tech.conv;
    if (news) conf += agreement * 7;
    const cap = news ? 97 : 85; // honest cap when running technical-only
    const confidence = clamp(round(conf, 0), 1, cap);

    const direction = finalNet > 0.1 ? "bullish" : finalNet < -0.1 ? "bearish" : "neutral";
    return {
      horizon: cfg.key,
      label: cfg.label,
      direction,
      action: actionFor(direction, confidence),
      confidence,
      pattern: tech.pattern,
      technicalScore: tech.score,
      newsScore: news ? round(news.score, 2) : null,
      agreement,
      readings: tech.readings,
    };
  });
}
