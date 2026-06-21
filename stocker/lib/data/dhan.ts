/**
 * Dhan OHLCV + quote provider (server-only) — NSE equities only. Dhan keys off a
 * numeric securityId, so `dhan-nse-map.json` maps our display tickers → securityId
 * (built from Dhan's scrip master). Daily uses the historical endpoint; intraday uses
 * the intraday endpoint (1h/4h from 60-min, 15m native); live snapshots use the batch
 * marketfeed/quote endpoint. Enabled when a token is available and NSE_DATA_SOURCE !==
 * "yahoo"; callers fall back to Yahoo on empty/failure.
 *
 * All requests pass through a global ~5/sec rate gate — Dhan caps Data APIs at 5 req/sec
 * and an unthrottled concurrent scan was tripping it (empty 200s → false "no data").
 */
import "server-only";
import type { Bar, Timeframe } from "../engine/types";
import { getDhanToken } from "@/lib/server/dhanToken";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import nseMap from "./dhan-nse-map.json";

const MAP = nseMap as Record<string, string>;
const ROOT = "https://api.dhan.co/v2";

export function dhanEnabled(): boolean {
  return (process.env.NSE_DATA_SOURCE ?? "dhan").toLowerCase() !== "yahoo";
}

// Live scrip overlay: the refresh cron stores a fresh ticker→securityId map in Supabase
// (renamed/new listings). It's merged over the static JSON and cached in memory. Lookups
// stay synchronous; callers `await ensureScripMap()` once before a scan to populate it.
let overlay: Record<string, string> | null = null;
let overlayAt = 0;
const OVERLAY_TTL = 6 * 3600 * 1000;

export function clearScripOverlay() {
  overlay = null;
  overlayAt = 0;
}

export async function ensureScripMap(): Promise<void> {
  if (overlay && Date.now() - overlayAt < OVERLAY_TTL) return;
  try {
    const sb = getSupabaseAdmin();
    const { data } = await sb.from("stocker_dhan_scrip").select("map").eq("id", 1).maybeSingle();
    const m = (data?.map ?? null) as Record<string, string> | null;
    overlay = m && Object.keys(m).length ? { ...MAP, ...m } : { ...MAP };
  } catch {
    overlay = { ...MAP };
  }
  overlayAt = Date.now();
}

/** securityId for an NSE display ticker (e.g. "RELIANCE" → "2885"), or null if unmapped. */
export function dhanSecurityId(displaySymbol: string): string | null {
  const m = overlay ?? MAP;
  return m[displaySymbol] ?? m[displaySymbol.toUpperCase()] ?? null;
}

// Lightweight concurrency cap on in-flight Dhan calls. Keeps the scan parallel (fast)
// while bounding bursts so we don't fling hundreds of requests at once. Tunable via
// DHAN_MAX_CONCURRENCY. NOT a hard 5/sec serializer — that made big scans ~4x slower for
// no real gain (the "empties" it chased were the liquidity floor, not throttling); any
// occasional 429 simply falls back to Yahoo.
const MAX_INFLIGHT = Math.max(1, Number(process.env.DHAN_MAX_CONCURRENCY) || 8);
let inflight = 0;
const waiters: Array<() => void> = [];
async function acquire(): Promise<void> {
  if (inflight < MAX_INFLIGHT) {
    inflight++;
    return;
  }
  // Wait for a slot; release() hands this waiter the slot directly (no re-increment).
  await new Promise<void>((res) => waiters.push(res));
}
function release(): void {
  const next = waiters.shift();
  if (next) next();
  else inflight--;
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

/** POST to a Dhan v2 endpoint (concurrency-capped). Returns parsed JSON or null. */
async function dhanPost<T>(path: string, body: Record<string, unknown>): Promise<T | null> {
  const token = await getDhanToken();
  if (!token) return null;
  await acquire();
  try {
    const res = await fetch(`${ROOT}/${path}`, {
      method: "POST",
      headers: { "access-token": token, "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    release();
  }
}

/** GET a Dhan v2 endpoint (concurrency-capped; read-only portfolio APIs — no IP needed). */
export async function dhanGet<T>(path: string): Promise<T | null> {
  const token = await getDhanToken();
  if (!token) return null;
  await acquire();
  try {
    const res = await fetch(`${ROOT}/${path}`, {
      headers: { "access-token": token, Accept: "application/json" },
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  } finally {
    release();
  }
}

export async function fetchDhanOhlcv(securityId: string, tf: Timeframe, lookbackBars = 400): Promise<Bar[]> {
  const now = Date.now();
  const seg = { securityId, exchangeSegment: "NSE_EQ", instrument: "EQUITY" };

  if (tf === "1d") {
    const fromMs = now - Math.max(lookbackBars * 2, 400) * 86400000;
    return toBars(await dhanPost<DhanCandles>("charts/historical", { ...seg, expiryCode: 0, fromDate: ymd(fromMs), toDate: ymd(now) }));
  }

  // Intraday: 15m native; 1h and 4h from 60-min bars (4h resampled).
  const interval = tf === "15m" ? "15" : "60";
  const days = tf === "15m" ? 60 : 90;
  const fromMs = now - days * 86400000;
  const bars = toBars(await dhanPost<DhanCandles>("charts/intraday", { ...seg, interval, fromDate: ymd(fromMs), toDate: ymd(now) }));
  return tf === "4h" ? resample4h(bars) : bars;
}

// ── Live batch quotes (marketfeed/quote) ──────────────────────────────────────

export interface DhanQuote {
  securityId: string;
  lastPrice: number;
  open: number;
  high: number;
  low: number;
  close: number; // previous close
  volume: number; // today cumulative
  upperCircuit: number | null;
  lowerCircuit: number | null;
}

interface QuoteEntry {
  last_price?: number;
  volume?: number;
  ohlc?: { open?: number; high?: number; low?: number; close?: number };
  upper_circuit_limit?: number;
  lower_circuit_limit?: number;
}
interface QuoteResponse {
  data?: { NSE_EQ?: Record<string, QuoteEntry> };
  status?: string;
}

/**
 * Live snapshot for many securityIds in one call (Dhan allows up to 1000/req at 1/sec).
 * Returns a map keyed by securityId. Missing ids = not trading (delisted/halted).
 */
export async function fetchDhanQuotes(securityIds: string[]): Promise<Map<string, DhanQuote>> {
  const out = new Map<string, DhanQuote>();
  const ids = [...new Set(securityIds.filter(Boolean))];
  for (let i = 0; i < ids.length; i += 1000) {
    const chunk = ids.slice(i, i + 1000).map((s) => Number(s)).filter((n) => Number.isFinite(n));
    if (!chunk.length) continue;
    const j = await dhanPost<QuoteResponse>("marketfeed/quote", { NSE_EQ: chunk });
    const rows = j?.data?.NSE_EQ;
    if (!rows) continue;
    for (const [sid, q] of Object.entries(rows)) {
      out.set(sid, {
        securityId: sid,
        lastPrice: q.last_price ?? 0,
        open: q.ohlc?.open ?? 0,
        high: q.ohlc?.high ?? 0,
        low: q.ohlc?.low ?? 0,
        close: q.ohlc?.close ?? 0,
        volume: q.volume ?? 0,
        upperCircuit: q.upper_circuit_limit ?? null,
        lowerCircuit: q.lower_circuit_limit ?? null,
      });
    }
  }
  return out;
}

/** True when the last price is pinned at the day's circuit band (untradeable that side). */
export function isCircuitLocked(q: DhanQuote): "upper" | "lower" | null {
  const p = q.lastPrice;
  if (!p) return null;
  if (q.upperCircuit && p >= q.upperCircuit - 1e-6) return "upper";
  if (q.lowerCircuit && p <= q.lowerCircuit + 1e-6) return "lower";
  return null;
}
