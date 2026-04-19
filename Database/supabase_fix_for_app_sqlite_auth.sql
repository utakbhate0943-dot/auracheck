-- Fix Supabase schema to match current app.py write model (SQLite auth + Supabase mirror)
-- Run in Supabase SQL Editor.

-- 1) Remove FK to auth.users so app-managed UUIDs can be inserted into public.users.
alter table if exists public.users
  drop constraint if exists users_user_id_fkey;

-- 2) Ensure columns used by app upsert exist on public.users.
alter table if exists public.users
  add column if not exists password_hash text,
  add column if not exists password_salt text,
  add column if not exists phone_number text,
  add column if not exists city text,
  add column if not exists zip_code text,
  add column if not exists is_verified boolean default false;

-- Backfill nullable rows before NOT NULL enforcement.
update public.users set password_hash = coalesce(password_hash, '') where password_hash is null;
update public.users set password_salt = coalesce(password_salt, '') where password_salt is null;

alter table if exists public.users
  alter column password_hash set not null,
  alter column password_salt set not null,
  alter column is_verified set default false;

-- 3) Keep/ensure useful uniqueness constraints.
create unique index if not exists users_user_id_uidx on public.users (user_id);
create unique index if not exists users_email_uidx on public.users (email);

-- 4) Allow anonymous signup inserts for the current app flow.
drop policy if exists users_insert_anon on public.users;
create policy users_insert_anon
on public.users
for insert
to anon
with check (true);

-- 5) Ensure profile/daily_inputs constraints expected by app.
create unique index if not exists profile_user_id_uidx on public.profile (user_id);
create unique index if not exists daily_inputs_user_date_uidx on public.daily_inputs (user_id, input_date);

-- 6) Align the cluster constraint with the 4-cluster KMeans model.
alter table if exists public.daily_inputs
  drop constraint if exists daily_inputs_cluster_check;

alter table if exists public.daily_inputs
  add constraint daily_inputs_cluster_check
  check (cluster is null or cluster between 0 and 3);
