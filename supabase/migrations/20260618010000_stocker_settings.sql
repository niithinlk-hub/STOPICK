-- Per-user LLM / news settings for the commodities predictor. The API key is
-- private to each user via RLS; the server reads it under the caller's session.
create table if not exists public.stocker_settings (
  user_id      uuid primary key default auth.uid() references auth.users (id) on delete cascade,
  llm_base_url text,
  llm_model    text,
  llm_api_key  text,
  news_enabled boolean not null default true,
  updated_at   timestamptz not null default now()
);

alter table public.stocker_settings enable row level security;

drop policy if exists "settings owner access" on public.stocker_settings;
create policy "settings owner access" on public.stocker_settings
  for all to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
