/**
 * Offline calibration harness — runs validateEngine on this host (Yahoo not IP-throttled
 * here, unlike Railway) so weights/exits can be fit with a clean, reproducible A/B.
 * Not shipped to the app. Run: npx tsx scripts/optimize.ts
 */
import type { Bar, Market, ScoringProfile } from "../lib/engine/types";
import { validateEngine } from "../lib/engine/validate";
import { CONFIG } from "../lib/config";

const HEADERS = {
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
  Accept: "application/json,text/plain,*/*",
};

const NSE = [
  "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL","ITC","KOTAKBANK",
  "LT","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI","SUNPHARMA","TITAN","ULTRACEMCO","WIPRO","NESTLEIND",
  "ONGC","NTPC","POWERGRID","M&M","TATAMOTORS","TATASTEEL","JSWSTEEL","ADANIENT","ADANIPORTS","COALINDIA",
  "GRASIM","HCLTECH","TECHM","BAJAJFINSV","DRREDDY","CIPLA","EICHERMOT","BRITANNIA","HINDALCO","BPCL",
  "INDUSINDBK","SBILIFE","HDFCLIFE","DABUR","PIDILITIND","HAVELLS","SIEMENS","DLF","TRENT","VEDL",
];

async function fetchBars(symbol: string): Promise<Bar[]> {
  const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=3y&interval=1d`;
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch(url, { headers: HEADERS });
      if (!res.ok) { await new Promise((r) => setTimeout(r, 400 * (attempt + 1))); continue; }
      const j: any = await res.json();
      const r = j?.chart?.result?.[0];
      const ts: number[] = r?.timestamp ?? [];
      const q = r?.indicators?.quote?.[0] ?? {};
      const bars: Bar[] = [];
      for (let i = 0; i < ts.length; i++) {
        const o = q.open?.[i], h = q.high?.[i], l = q.low?.[i], c = q.close?.[i];
        if (o == null || h == null || l == null || c == null) continue;
        bars.push({ time: ts[i], open: o, high: h, low: l, close: c, volume: q.volume?.[i] ?? 0 });
      }
      if (bars.length) return bars;
    } catch { await new Promise((r) => setTimeout(r, 400 * (attempt + 1))); }
  }
  return [];
}

async function mapLimit<T, R>(items: T[], limit: number, fn: (t: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let i = 0;
  await Promise.all(Array.from({ length: limit }, async () => {
    while (i < items.length) { const k = i++; out[k] = await fn(items[k]); }
  }));
  return out;
}

const BASE = CONFIG.scoringProfiles.bullish_breakout;
const NEW_WEIGHTS: Record<string, number> = {
  trend_alignment: 6, structure_quality: 4, breakout_quality: 16, pullback_quality: 2,
  volume_confirmation: 18, momentum: 12, volatility_regime: 8, relative_strength: 9,
  liquidity: 8, htf_headroom: 3, rr_ratio: 3, market_regime: 4, mtf_agreement: 3,
  follow_through: 13, event_risk: 3, index_alignment: 6,
};
const NEW_THRESHOLDS = { "A+": 86, A: 80, B: 73, C: 65 };

// Conservative, cross-validated tweak: bump the ONE component positive in both markets
// (relative_strength), fund it by trimming the measured-flat trend stack. Everything else
// stays at the live baseline — deliberately NOT a 16-dim refit (that overfits noise).
const RS_WEIGHTS: Record<string, number> = {
  ...BASE.weights as any,
  relative_strength: 16, index_alignment: 6,
  trend_alignment: 9, structure_quality: 8, mtf_agreement: 4, htf_headroom: 4,
};

function profile(weights: Record<string, number>, thresholds: Record<string, number>): ScoringProfile {
  return { ...BASE, weights: weights as any, gradeThresholds: thresholds as any };
}

const US = [
  "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AVGO","JPM","V",
  "UNH","XOM","LLY","JNJ","WMT","MA","PG","HD","COST","ORCL",
  "MRK","ABBV","CVX","CRM","AMD","NFLX","KO","PEP","ADBE","BAC",
  "TMO","CSCO","ACN","MCD","ABT","WFC","DIS","QCOM","INTC","TXN",
  "CAT","GE","BA","NKE","HON","UNP","LOW","INTU","AMAT","NOW",
];

// Phase 2 (2026-06-26): universe + stop-geometry test. Weights are already re-fit and LIVE
// (RS 16) — here BASE = live config. Questions: (a) do mid-cap tier_2 breakouts carry more
// edge than tier_1 large caps? (b) does a wider/tighter ATR stop beat the plan stop?
// (c) do the realized score-bins justify moving grade thresholds?
import { UNIVERSES } from "../lib/data/universe";

/** Evenly-strided sample so we span the whole tier list, not just its (cap-ordered) head. */
function sample(list: string[], n: number): string[] {
  if (list.length <= n) return [...list];
  const step = list.length / n;
  return Array.from({ length: n }, (_, i) => list[Math.floor(i * step)]);
}

const fmt = (r: any) =>
  `trades=${r.overall.trades} spearman=${r.spearman} expR=${r.overall.expectancyR} win=${r.overall.winRate}% ` +
  `hit=${r.overall.targetHitRate}% byGrade=[${r.byGrade.map((g: any) => `${g.grade}:${g.expectancyR}(${g.trades})`).join(" ")}]`;

async function runUniverse(market: Market, label: string, syms: string[], benchSym: string, suffix: string) {
  console.log(`\n######## ${market} ${label} (${syms.length} names) ########`);
  const bench = await fetchBars(benchSym);
  const fetched = await mapLimit(syms, 6, async (s) => ({ symbol: s, bars: await fetchBars(`${s}${suffix}`) }));
  const items = fetched.filter((x) => x.bars.length >= 300);
  console.log(`bench ${benchSym} bars=${bench.length}, symbols >=300 bars: ${items.length}/${syms.length}`);

  // Cap test: structural plan stop, but depth capped at k×ATR (only binds on deep base lows).
  const planStop = validateEngine(items, bench, market, { horizonBars: 20, targetR: 2 });
  const cap25 = validateEngine(items, bench, market, { horizonBars: 20, targetR: 2, stopAtrCap: 2.5 });
  const cap3 = validateEngine(items, bench, market, { horizonBars: 20, targetR: 2, stopAtrCap: 3 });
  console.log("  planStop:", fmt(planStop));
  console.log("  cap2.5  :", fmt(cap25));
  console.log("  cap3.0  :", fmt(cap3));
}

async function main() {
  const nseT2 = sample(UNIVERSES.NSE.tier_2, 60).map((s) => s.replace(/\.NS$/, ""));
  const usT2 = sample(UNIVERSES.US.tier_2, 60);
  await runUniverse("NSE", "tier_1", NSE, "^NSEI", ".NS");
  await runUniverse("NSE", "tier_2", nseT2, "^NSEI", ".NS");
  await runUniverse("US", "tier_1", US, "SPY", "");
  await runUniverse("US", "tier_2", usT2, "SPY", "");
}

main().catch((e) => { console.error(e); process.exit(1); });
