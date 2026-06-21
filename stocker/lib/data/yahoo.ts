/**
 * Yahoo Finance OHLCV provider (server-only). Hits Yahoo's public v8 chart
 * endpoint directly via fetch — the same data source STOPICK's yfinance used —
 * producing the unified Bar schema. 4h bars are resampled from 60m, matching
 * STOPICK's loader.
 */
import "server-only";
import type { Bar, Timeframe } from "../engine/types";
import { cacheGet, cacheSet } from "./cache";

const HEADERS = {
  "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
  Accept: "application/json,text/plain,*/*",
};

type YahooInterval = "1d" | "60m" | "15m" | "5m";

function providerInterval(tf: Timeframe): YahooInterval {
  if (tf === "4h" || tf === "1h") return "60m";
  if (tf === "15m") return "15m";
  return "1d";
}

/** Lookback start in epoch seconds — mirrors STOPICK _start_for_interval. */
function startSeconds(tf: Timeframe, lookbackBars: number): number {
  const now = Math.floor(Date.now() / 1000);
  const day = 86400;
  if (tf === "1d") return now - Math.max(lookbackBars * 2, 365) * day;
  if (tf === "4h" || tf === "1h") return now - Math.max(Math.floor(lookbackBars / 6), 180) * day;
  if (tf === "15m") return now - Math.max(Math.floor(lookbackBars / 20), 55) * day;
  return now - Math.max(Math.floor(lookbackBars / 40), 28) * day;
}

function resample4h(bars: Bar[]): Bar[] {
  if (!bars.length) return bars;
  const buckets = new Map<number, Bar>();
  const order: number[] = [];
  const span = 4 * 3600;
  for (const b of bars) {
    const key = Math.floor(b.time / span) * span;
    const existing = buckets.get(key);
    if (!existing) {
      buckets.set(key, { time: key, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume });
      order.push(key);
    } else {
      existing.high = Math.max(existing.high, b.high);
      existing.low = Math.min(existing.low, b.low);
      existing.close = b.close;
      existing.volume += b.volume;
    }
  }
  return order.map((k) => buckets.get(k)!);
}

interface YahooChartResponse {
  chart?: {
    result?: Array<{
      timestamp?: number[];
      indicators?: {
        quote?: Array<{
          open?: (number | null)[];
          high?: (number | null)[];
          low?: (number | null)[];
          close?: (number | null)[];
          volume?: (number | null)[];
        }>;
      };
    }>;
    error?: { code?: string; description?: string } | null;
  };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Fetch + parse one chart, with retry/backoff and a query1→query2 host fallback.
 * Yahoo rate-limits a server IP under bursty scans (popular tickers are edge-cached
 * and sail through; thinly-traded small-caps get 429'd or return an empty 200), so
 * we retry transient failures on the alternate host before giving up.
 */
async function fetchChartBars(
  symbol: string,
  interval: YahooInterval,
  period1: number,
  period2: number,
): Promise<Bar[]> {
  const path =
    `/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${period1}&period2=${period2}` +
    `&interval=${interval}&includePrePost=false&events=div%2Csplits`;

  for (let attempt = 0; attempt < 3; attempt++) {
    const host = attempt === 0 ? "query1" : "query2";
    try {
      const res = await fetch(`https://${host}.finance.yahoo.com${path}`, { headers: HEADERS, cache: "no-store" });
      if (res.status === 429 || res.status >= 500) {
        await sleep(300 * (attempt + 1) + Math.floor(Math.random() * 200));
        continue;
      }
      if (!res.ok) return [];
      const data = (await res.json()) as YahooChartResponse;
      const result = data.chart?.result?.[0];
      const ts = result?.timestamp ?? [];
      const q = result?.indicators?.quote?.[0];
      const bars: Bar[] = [];
      if (q && ts.length) {
        for (let i = 0; i < ts.length; i++) {
          const o = q.open?.[i];
          const h = q.high?.[i];
          const l = q.low?.[i];
          const c = q.close?.[i];
          if (o == null || h == null || l == null || c == null) continue;
          bars.push({ time: ts[i], open: o, high: h, low: l, close: c, volume: q.volume?.[i] ?? 0 });
        }
      }
      if (bars.length) return bars;
      // Empty 200 — often a disguised throttle; brief backoff then retry on query2.
      await sleep(250 * (attempt + 1));
    } catch {
      await sleep(250 * (attempt + 1));
    }
  }
  return [];
}

export async function fetchOhlcv(symbol: string, tf: Timeframe, lookbackBars = 400): Promise<Bar[]> {
  const cacheKey = `ohlcv|${symbol}|${tf}|${lookbackBars}`;
  const cached = cacheGet<Bar[]>(cacheKey);
  if (cached) return cached;

  const interval = providerInterval(tf);
  const period1 = startSeconds(tf, lookbackBars);
  const period2 = Math.floor(Date.now() / 1000);

  let bars = await fetchChartBars(symbol, interval, period1, period2);
  if (tf === "4h") bars = resample4h(bars);
  // Only cache successful fetches so a throttled empty isn't pinned for the TTL.
  if (bars.length) cacheSet(cacheKey, bars);
  return bars;
}

/**
 * Best-effort next-earnings proximity in days (null when unavailable). The
 * quoteSummary endpoint requires a crumb/cookie and is unreliable from a server,
 * so this fails closed to null — the engine treats null as "no event risk known".
 */
export async function fetchEventDays(symbol: string): Promise<number | null> {
  const cacheKey = `event|${symbol}`;
  const cached = cacheGet<number | null>(cacheKey);
  if (cached !== undefined) return cached;

  let days: number | null = null;
  try {
    const url =
      `https://query1.finance.yahoo.com/v10/finance/quoteSummary/${encodeURIComponent(symbol)}` +
      `?modules=calendarEvents`;
    const res = await fetch(url, { headers: HEADERS, cache: "no-store" });
    if (res.ok) {
      const data = (await res.json()) as {
        quoteSummary?: { result?: Array<{ calendarEvents?: { earnings?: { earningsDate?: Array<{ raw?: number }> } } }> };
      };
      const raw = data.quoteSummary?.result?.[0]?.calendarEvents?.earnings?.earningsDate?.[0]?.raw;
      if (raw) days = Math.round((raw * 1000 - Date.now()) / 86400000);
    }
  } catch {
    days = null;
  }
  cacheSet(cacheKey, days);
  return days;
}
