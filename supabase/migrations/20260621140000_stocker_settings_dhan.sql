-- Dhan access token (NSE data). Short-lived (~24h), so it's stored per user and
-- refreshable from the app or via the bot's /access command, overriding env.
alter table public.stocker_settings add column if not exists dhan_token text;
