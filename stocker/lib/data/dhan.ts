/**
 * Dhan OHLCV provider (server-only) — NSE equities only. Dhan keys off a numeric
 * securityId, so `dhan-nse-map.json` maps our display tickers → securityId (built
 * from Dhan's scrip master). Daily uses the historical endpoint; intraday uses the
 * intraday endpoint (1h/4h from 60-min, 15m native). Enabled when DHAN_ACCESS_TOKEN
 * is set and NSE_DATA_SOURCE !== "yahoo"; callers fall back to Yahoo on empty/failure.
 */
import "server-only";
import type { Bar, Timeframe } from "../engine/types";
import { getDhanToken } from "@/lib/server/dhanToken";
import nseMap from "./dhan-nse-map.json";

const MAP = nseMap as Record<string, string>;
const BASE = "https://api.dhan.co/v2/charts";

export function dhanEnabled(): boolean {
  return (process.env.NSE_DATA_SOURCE ?? "dhan").toLowerCase() !== "yahoo";
}

/** securityId for an NSE display ticker (e.g. "RELIANCE" → "2885"), or null if unmapped. */
export function dhanSecurityId(displaySymbol: string): string | null {
  return MAP[displaySymbol] ?? MAP[displaySymbol.toUpperCase()] ?? null;
}

const ymd = (ms: number) => new Date(ms).toISOString().slice(0, 10);

interface DhanCandles {
  open?: number[];
  high?: number[];
  low?: number[];
  close?: number[];
  volume?: number[];
  timestamp?: number[];
}

function toBars(d: DhanCandles | null): Bar[] {
  const ts = d?.timestamp;
  if (!Array.isArray(ts) || !Array.isArray(d?.close)) return [];
  const bars: Bar[] = [];
  for (let i = 0; i < ts.length; i++) {
    const o = d.open?.[i], h = d.high?.[i], l = d.low?.[i], c = d.close?.[i];
    if (o == null || h == null || l == null || c == null) continue;
    bars.push({ time: Math.floor(ts[i]), open: +o, high: +h, low: +l, close: +c, volume: +(d.volume?.[i] ?? 0) });
  }
  return bars;
}

function resample4h(bars: Bar[]): Bar[] {
  if (!bars.length) return bars;
  const buckets = new Map<number, Bar>();
  const order: number[] = [];
  const span = 4 * 3600;
  for (const b of bars) {
    const key = Math.floor(b.time / span) * span;
    const ex = buckets.get(key);
    if (!ex) {
      buckets.set(key, { time: key, open: b.open, high: b.high, low: b.low, close: b.close, volume: b.volume });
      order.push(key);
    } else {
      ex.high = Math.max(ex.high, b.high);
      ex.low = Math.min(ex.low, b.low);
      ex.close = b.close;
      ex.volume += b.volume;
    }
  }
  return order.map((k) => buckets.get(k)!);
}

async function call(path: "historical" | "intraday", body: Record<string, unknown>): Promise<DhanCandles | null> {
  const token = await getDhanToken();
  if (!token) return null;
  try {
    const res = await fetch(`${BASE}/${path}`, {
      method: "POST",
      headers: { "access-token": token, "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as DhanCandles;
  } catch {
    return null;
  }
}

export async function fetchDhanOhlcv(securityId: string, tf: Timeframe, lookbackBars = 400): Promise<Bar[]> {
  const now = Date.now();
  const seg = { securityId, exchangeSegment: "NSE_EQ", instrument: "EQUITY" };

  if (tf === "1d") {
    const fromMs = now - Math.max(lookbackBars * 2, 400) * 86400000;
    return toBars(await call("historical", { ...seg, expiryCode: 0, fromDate: ymd(fromMs), toDate: ymd(now) }));
  }

  // Intraday: 15m native; 1h and 4h from 60-min bars (4h resampled).
  const interval = tf === "15m" ? "15" : "60";
  const days = tf === "15m" ? 60 : 90;
  const fromMs = now - days * 86400000;
  const bars = toBars(await call("intraday", { ...seg, interval, fromDate: ymd(fromMs), toDate: ymd(now) }));
  return tf === "4h" ? resample4h(bars) : bars;
}
