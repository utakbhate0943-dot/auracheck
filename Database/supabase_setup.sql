-- Supabase setup for AuraCheck
-- Run in Supabase SQL Editor as project owner.

create extension if not exists pgcrypto;
create extension if not exists citext;

-- Keep timestamps consistent
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- 1) Users table (one row per auth user)
-- Passwords are NOT stored here. Supabase Auth handles password security.
create table if not exists public.users (
  user_id uuid primary key references auth.users(id) on delete cascade,
  email citext not null unique,
  first_name text not null check (char_length(trim(first_name)) between 1 and 80),
  last_name text not null check (char_length(trim(last_name)) between 1 and 80),
  phone_number text null check (phone_number is null or phone_number ~ '^[0-9+()\-\s]{7,20}$'),
  city text null check (city is null or char_length(trim(city)) <= 120),
  zip_code text null check (zip_code is null or char_length(trim(zip_code)) <= 20),
  is_verified boolean not null default false,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create trigger trg_users_updated_at
before update on public.users
for each row
execute procedure public.set_updated_at();

-- 2) Profile table (one profile per user)
create table if not exists public.profile (
  profile_id bigint generated always as identity primary key,
  user_id uuid not null unique references public.users(user_id) on delete cascade,
  age int null check (age is null or age between 10 and 120),
  lifestyle_parameters jsonb not null default '{}'::jsonb,
  personal_details jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  check (jsonb_typeof(lifestyle_parameters) = 'object'),
  check (jsonb_typeof(personal_details) = 'object')
);

create trigger trg_profile_updated_at
before update on public.profile
for each row
execute procedure public.set_updated_at();

create index if not exists idx_profile_user_id on public.profile(user_id);

-- 3) Daily inputs (one submission per user per day)
create table if not exists public.daily_inputs (
  entry_id bigint generated always as identity primary key,
  user_id uuid not null references public.users(user_id) on delete cascade,
  input_date date not null,
  submitted_at timestamptz not null default now(),
  answers_json jsonb not null,
  prediction_json jsonb not null,
  cluster int null check (cluster is null or cluster between 0 and 2),
  created_at timestamptz not null default now(),
  check (jsonb_typeof(answers_json) = 'object'),
  check (jsonb_typeof(prediction_json) = 'object'),
  unique (user_id, input_date)
);

create index if not exists idx_daily_inputs_user_date on public.daily_inputs(user_id, input_date desc);
create index if not exists idx_daily_inputs_submitted_at on public.daily_inputs(submitted_at desc);

-- Optional helper to auto-create public.users row after auth signup
create or replace function public.handle_new_auth_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.users (
    user_id,
    email,
    first_name,
    last_name,
    phone_number,
    city,
    zip_code,
    is_verified
  ) values (
    new.id,
    new.email,
    coalesce(new.raw_user_meta_data ->> 'first_name', 'First'),
    coalesce(new.raw_user_meta_data ->> 'last_name', 'Last'),
    nullif(new.raw_user_meta_data ->> 'phone_number', ''),
    nullif(new.raw_user_meta_data ->> 'city', ''),
    nullif(new.raw_user_meta_data ->> 'zip_code', ''),
    coalesce(new.email_confirmed_at is not null, false)
  )
  on conflict (user_id) do update
  set
    email = excluded.email,
    first_name = excluded.first_name,
    last_name = excluded.last_name,
    phone_number = excluded.phone_number,
    city = excluded.city,
    zip_code = excluded.zip_code,
    is_verified = excluded.is_verified,
    updated_at = now();

  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
after insert on auth.users
for each row execute procedure public.handle_new_auth_user();

-- Keep verification flag synced if auth email status changes
create or replace function public.sync_user_verification()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  update public.users
  set
    is_verified = coalesce(new.email_confirmed_at is not null, false),
    updated_at = now()
  where user_id = new.id;
  return new;
end;
$$;

drop trigger if exists on_auth_user_updated on auth.users;
create trigger on_auth_user_updated
after update of email_confirmed_at on auth.users
for each row execute procedure public.sync_user_verification();

-- Row Level Security
alter table public.users enable row level security;
alter table public.profile enable row level security;
alter table public.daily_inputs enable row level security;

-- Users policies: user sees and updates only own row; admins can read all.
drop policy if exists users_select_own on public.users;
create policy users_select_own
on public.users
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists users_insert_own on public.users;
create policy users_insert_own
on public.users
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists users_update_own on public.users;
create policy users_update_own
on public.users
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists users_admin_read_all on public.users;
create policy users_admin_read_all
on public.users
for select
to authenticated
using ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin');

-- Profile policies
drop policy if exists profile_select_own on public.profile;
create policy profile_select_own
on public.profile
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists profile_insert_own on public.profile;
create policy profile_insert_own
on public.profile
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists profile_update_own on public.profile;
create policy profile_update_own
on public.profile
for update
to authenticated
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

drop policy if exists profile_admin_read_all on public.profile;
create policy profile_admin_read_all
on public.profile
for select
to authenticated
using ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin');

-- Daily input policies (insert once per day is enforced by unique constraint)
drop policy if exists daily_inputs_select_own on public.daily_inputs;
create policy daily_inputs_select_own
on public.daily_inputs
for select
to authenticated
using (auth.uid() = user_id);

drop policy if exists daily_inputs_insert_own on public.daily_inputs;
create policy daily_inputs_insert_own
on public.daily_inputs
for insert
to authenticated
with check (auth.uid() = user_id);

drop policy if exists daily_inputs_admin_read_all on public.daily_inputs;
create policy daily_inputs_admin_read_all
on public.daily_inputs
for select
to authenticated
using ((auth.jwt() -> 'app_metadata' ->> 'role') = 'admin');

-- Optional: block updates/deletes for immutable daily history
revoke update, delete on public.daily_inputs from authenticated;

-- Helpful analytics view for admin dashboard
create or replace view public.v_daily_trend as
select
  input_date,
  count(*) as submissions,
  avg((prediction_json ->> 'stress_level')::numeric) as avg_stress,
  avg((prediction_json ->> 'anxiety_score')::numeric) as avg_anxiety,
  avg((prediction_json ->> 'depression_score')::numeric) as avg_depression
from public.daily_inputs
group by input_date
order by input_date;
