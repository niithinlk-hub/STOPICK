import "server-only";
import type { Headline } from "@/lib/data/news";

export interface LlmConfig {
  baseUrl: string;
  model: string;
  apiKey: string;
}

export interface NewsAnalysis {
  score: number; // -1 (very bearish) .. +1 (very bullish)
  label: string;
  summary: string;
  drivers: string[];
}

const SYSTEM =
  "You are a commodities market analyst. Given recent news headlines for a commodity, judge the net " +
  "directional impact on its price. Respond ONLY with compact JSON: " +
  '{"score": <number -1..1>, "label": "<one of: Strongly Bearish, Bearish, Neutral, Bullish, Strongly Bullish>", ' +
  '"summary": "<=2 sentences", "drivers": ["<short driver>", ...up to 4]}. ' +
  "score is the expected near-term price impact: positive = upward pressure. Be objective and concise.";

/** Ask an OpenAI-compatible chat endpoint to score news sentiment for a commodity. */
export async function analyzeNews(
  commodity: string,
  headlines: Headline[],
  cfg: LlmConfig,
  signal?: AbortSignal,
): Promise<NewsAnalysis> {
  const list = headlines.map((h, i) => `${i + 1}. ${h.title}${h.source ? ` (${h.source})` : ""}`).join("\n");
  const base = cfg.baseUrl.replace(/\/+$/, "");
  const res = await fetch(`${base}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.apiKey}` },
    body: JSON.stringify({
      model: cfg.model,
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM },
        { role: "user", content: `Commodity: ${commodity}\nRecent headlines:\n${list}\n\nReturn the JSON now.` },
      ],
    }),
    signal,
  });

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`LLM error ${res.status}: ${body.slice(0, 200)}`);
  }
  const data = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  const content = data.choices?.[0]?.message?.content ?? "";
  const parsed = extractJson(content);

  const score = clampNum(parsed.score, -1, 1);
  return {
    score,
    label: typeof parsed.label === "string" ? parsed.label : labelFor(score),
    summary: typeof parsed.summary === "string" ? parsed.summary : "",
    drivers: Array.isArray(parsed.drivers) ? parsed.drivers.map(String).slice(0, 4) : [],
  };
}

function extractJson(text: string): Record<string, unknown> {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/);
  const raw = fenced ? fenced[1] : text;
  const start = raw.indexOf("{");
  const end = raw.lastIndexOf("}");
  if (start === -1 || end === -1) return {};
  try {
    return JSON.parse(raw.slice(start, end + 1)) as Record<string, unknown>;
  } catch {
    return {};
  }
}

function clampNum(v: unknown, lo: number, hi: number): number {
  const n = typeof v === "number" ? v : Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.max(lo, Math.min(hi, n));
}

function labelFor(score: number): string {
  if (score >= 0.5) return "Strongly Bullish";
  if (score >= 0.15) return "Bullish";
  if (score <= -0.5) return "Strongly Bearish";
  if (score <= -0.15) return "Bearish";
  return "Neutral";
}
