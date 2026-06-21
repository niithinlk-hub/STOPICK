import "server-only";
import { createSupabaseServer } from "@/lib/supabase/server";

export interface UserSettings {
  llmBaseUrl: string | null;
  llmModel: string | null;
  llmApiKey: string | null;
  newsEnabled: boolean;
}

/** Read the signed-in user's LLM/news settings (RLS scopes to their own row). */
export async function getUserSettings(): Promise<UserSettings | null> {
  const supabase = createSupabaseServer();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  const { data } = await supabase
    .from("stocker_settings")
    .select("llm_base_url, llm_model, llm_api_key, news_enabled")
    .eq("user_id", user.id)
    .maybeSingle();

  if (!data) return { llmBaseUrl: null, llmModel: null, llmApiKey: null, newsEnabled: true };
  return {
    llmBaseUrl: data.llm_base_url,
    llmModel: data.llm_model,
    llmApiKey: data.llm_api_key,
    newsEnabled: data.news_enabled,
  };
}
