import { NextResponse } from "next/server";
import { cronAuthorized } from "@/lib/server/cronAuth";
import { createSupabaseServer } from "@/lib/supabase/server";
import { getSupabaseAdmin } from "@/lib/supabase/admin";
import { isAdminEmail } from "@/lib/server/admins";
import { clearScripOverlay } from "@/lib/data/dhan";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const CSV_URL = "https://images.dhan.co/api-data/api-scrip-master-detailed.csv";

/** Minimal CSV line tokenizer that respects double-quoted fields (names with commas). */
function parseCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (q) {
      if (ch === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else q = false;
      } else cur += ch;
    } else if (ch === ",") {
      out.push(cur);
      cur = "";
    } else if (ch === '"') q = true;
    else cur += ch;
  }
  out.push(cur);
  return out;
}

async function authorized(req: Request): Promise<boolean> {
  if (cronAuthorized(req)) return true;
  try {
    const {
      data: { user },
    } = await createSupabaseServer().auth.getUser();
    return Boolean(user && isAdminEmail(user.email));
  } catch {
    return false;
  }
}

/**
 * Rebuild the NSE display-ticker → securityId map from Dhan's scrip master and store it
 * in Supabase (overlay over the static map). Keeps renamed/new listings mapped so they
 * stop falling back to Yahoo / showing "No scan frame data". Cron-weekly or admin-manual.
 */
export async function GET(req: Request) {
  if (!(await authorized(req))) return NextResponse.json({ error: "Unauthorized" }, { status: 401 });

  let text: string;
  try {
    const res = await fetch(CSV_URL, { cache: "no-store" });
    if (!res.ok) return NextResponse.json({ error: `scrip master ${res.status}` }, { status: 502 });
    text = await res.text();
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : "fetch failed" }, { status: 502 });
  }

  const lines = text.split(/\r?\n/);
  const header = parseCsvLine(lines[0]);
  const col = (name: string) => header.indexOf(name);
  const cExch = col("EXCH_ID"),
    cSeg = col("SEGMENT"),
    cSid = col("SECURITY_ID"),
    cInst = col("INSTRUMENT"),
    cSym = col("UNDERLYING_SYMBOL"),
    cSeries = col("SERIES");
  if ([cExch, cSeg, cSid, cInst, cSym, cSeries].some((i) => i < 0)) {
    return NextResponse.json({ error: "unexpected scrip master columns", header }, { status: 502 });
  }

  const map: Record<string, string> = {};
  for (let i = 1; i < lines.length; i++) {
    if (!lines[i]) continue;
    const c = parseCsvLine(lines[i]);
    if (c[cExch] !== "NSE" || c[cSeg] !== "E" || c[cInst] !== "EQUITY" || c[cSeries] !== "EQ") continue;
    const sym = (c[cSym] || "").trim().toUpperCase();
    const sid = (c[cSid] || "").trim();
    if (sym && sid) map[sym] = sid;
  }

  const count = Object.keys(map).length;
  if (count < 500) return NextResponse.json({ error: `parsed only ${count} names — refusing to overwrite`, count }, { status: 502 });

  try {
    const sb = getSupabaseAdmin();
    const { error } = await sb
      .from("stocker_dhan_scrip")
      .upsert({ id: 1, map, count, updated_at: new Date().toISOString() }, { onConflict: "id" });
    if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : "db error" }, { status: 500 });
  }

  clearScripOverlay();
  return NextResponse.json({ ok: true, count });
}
