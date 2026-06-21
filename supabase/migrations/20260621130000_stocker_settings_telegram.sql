-- Telegram digest preferences (enabled, sets, min score, top N, pre-close,
-- intraday) stored per user as JSON. The daily cron reads the admin's row via
-- the service role; the app edits it under the owner's session.
alter table public.stocker_settings add column if not exists telegram jsonb;
