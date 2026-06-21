-- Intraday alert de-dup log: one row per (user, alert_key) so the watchlist
-- intraday cron only pings each fresh trigger once per day, not every 20 minutes.
create table if not exists public.stocker_alerts (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null,
  alert_key text not null,
  created_at timestamptz not null default now()
);

create unique index if not exists stocker_alerts_user_key
  on public.stocker_alerts (user_id, alert_key);

alter table public.stocker_alerts enable row level security;

do $$
begin
  if not exists (
    select 1 from pg_policies
    where schemaname = 'public' and tablename = 'stocker_alerts'
      and policyname = 'stocker_alerts_owner_select'
  ) then
    create policy stocker_alerts_owner_select on public.stocker_alerts
      for select to authenticated using (auth.uid() = user_id);
  end if;
end $$;
