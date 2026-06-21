-- Live Dhan NSE scrip map (display ticker -> securityId), refreshed from Dhan's scrip
-- master so renamed/new listings stay mapped. Single row; service-role only (RLS on,
-- no policies = no anon/authenticated access). Read at runtime as an overlay over the
-- static dhan-nse-map.json shipped in the build.
create table if not exists public.stocker_dhan_scrip (
  id smallint primary key default 1,
  map jsonb not null default '{}'::jsonb,
  count integer not null default 0,
  updated_at timestamptz not null default now(),
  constraint stocker_dhan_scrip_single check (id = 1)
);

alter table public.stocker_dhan_scrip enable row level security;
