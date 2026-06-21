-- STOCKER: watchlist + paper-trading blotter, scoped per authenticated user.
-- Namespaced `stocker_*` so they don't collide with anything else in this project.

create table if not exists public.stocker_watchlist (
  id         uuid primary key default gen_random_uuid(),
  user_id    uuid not null default auth.uid() references auth.users (id) on delete cascade,
  symbol     text not null,
  market     text not null check (market in ('US', 'NSE')),
  note       text,
  created_at timestamptz not null default now(),
  unique (user_id, symbol, market)
);

create table if not exists public.stocker_trades (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null default auth.uid() references auth.users (id) on delete cascade,
  ticker       text not null,
  market       text not null check (market in ('US', 'NSE')),
  side         text not null default 'long',
  qty          numeric not null,
  entry        numeric not null,
  stop         numeric,
  target       numeric,
  opened_at    timestamptz not null default now(),
  setup_family text,
  score        numeric,
  grade        text,
  pattern      text,
  source       text not null default 'manual' check (source in ('scan', 'manual')),
  status       text not null default 'open' check (status in ('open', 'closed')),
  exit         numeric,
  closed_at    timestamptz,
  note         text
);

create index if not exists stocker_watchlist_user_idx on public.stocker_watchlist (user_id, created_at desc);
create index if not exists stocker_trades_user_idx on public.stocker_trades (user_id, opened_at desc);

-- Row Level Security: every user only ever sees / mutates their own rows.
alter table public.stocker_watchlist enable row level security;
alter table public.stocker_trades enable row level security;

drop policy if exists "watchlist owner access" on public.stocker_watchlist;
create policy "watchlist owner access" on public.stocker_watchlist
  for all to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);

drop policy if exists "trades owner access" on public.stocker_trades;
create policy "trades owner access" on public.stocker_trades
  for all to authenticated
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
