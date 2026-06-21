-- Biometric / passkey (WebAuthn) credentials for Stocker Analytics.
-- One row per registered authenticator (Touch ID / Face ID / Windows Hello / security key).
-- Writes happen server-side via the service_role (which bypasses RLS); the owner can
-- read and revoke their own credentials from the client.

create table if not exists public.stocker_passkeys (
  cred_id text primary key,
  user_id uuid not null references auth.users (id) on delete cascade,
  public_key text not null,
  counter bigint not null default 0,
  transports text[] not null default '{}',
  device_label text,
  created_at timestamptz not null default now(),
  last_used_at timestamptz
);

create index if not exists stocker_passkeys_user_id_idx on public.stocker_passkeys (user_id);

alter table public.stocker_passkeys enable row level security;

drop policy if exists "stocker_passkeys_select_own" on public.stocker_passkeys;
create policy "stocker_passkeys_select_own" on public.stocker_passkeys
  for select to authenticated using (auth.uid() = user_id);

drop policy if exists "stocker_passkeys_delete_own" on public.stocker_passkeys;
create policy "stocker_passkeys_delete_own" on public.stocker_passkeys
  for delete to authenticated using (auth.uid() = user_id);
