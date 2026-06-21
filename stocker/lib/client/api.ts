"use client";

import type {
  Market,
  ScanParams,
  ScanResponse,
  SetupFamily,
  SymbolDetailResponse,
  Timeframe,
} from "@/lib/engine/types";
import type { BacktestResult } from "@/lib/engine/backtest";
import type { CommoditiesResponse } from "@/lib/engine/commodities";
import type { Country, ConfluenceResponse } from "@/lib/engine/types";

export interface ConfluenceParams {
  country: Country;
  source: "tier_1" | "tier_2" | "sample" | "manual";
  timeframe: Timeframe;
  side: "long" | "short" | "both";
  manualSymbols?: string;
  limit?: number;
}

export async function runConfluence(params: ConfluenceParams, signal?: AbortSignal): Promise<ConfluenceResponse> {
  const res = await fetch("/api/confluence", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal,
  });
  return jsonOrThrow<ConfluenceResponse>(res);
}

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { error?: string };
    throw new Error(body.error ?? `Request failed with ${res.status}`);
  }
  return (await res.json()) as T;
}

export async function scanStocks(params: ScanParams, signal?: AbortSignal): Promise<ScanResponse> {
  const res = await fetch("/api/scan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
    signal,
  });
  return jsonOrThrow<ScanResponse>(res);
}

export async function getSymbolDetail(
  ticker: string,
  market: Market,
  timeframe: Timeframe,
  family: SetupFamily = "breakout",
  signal?: AbortSignal,
): Promise<SymbolDetailResponse> {
  const qs = new URLSearchParams({ market, timeframe, family });
  const res = await fetch(`/api/symbol/${encodeURIComponent(ticker)}?${qs}`, { signal });
  return jsonOrThrow<SymbolDetailResponse>(res);
}

export interface Quote {
  ticker: string;
  market: Market;
  price: number;
  prevClose: number;
  asOf: number;
}

export async function getQuote(ticker: string, market: Market, signal?: AbortSignal): Promise<Quote> {
  const qs = new URLSearchParams({ market });
  const res = await fetch(`/api/quote/${encodeURIComponent(ticker)}?${qs}`, { signal });
  return jsonOrThrow<Quote>(res);
}

export async function getCommodities(signal?: AbortSignal): Promise<CommoditiesResponse> {
  const res = await fetch("/api/commodities", { signal });
  return jsonOrThrow<CommoditiesResponse>(res);
}

export type BacktestApiResult = BacktestResult & { ticker: string; market: Market };

export async function getBacktest(ticker: string, market: Market, signal?: AbortSignal): Promise<BacktestApiResult> {
  const qs = new URLSearchParams({ market });
  const res = await fetch(`/api/backtest/${encodeURIComponent(ticker)}?${qs}`, { signal });
  return jsonOrThrow<BacktestApiResult>(res);
}
