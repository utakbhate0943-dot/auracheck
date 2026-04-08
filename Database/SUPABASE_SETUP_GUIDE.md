# AuraCheck Supabase Setup Guide (Database)

This guide is based on:
- `Database/supabase_setup.sql` (recommended Supabase schema)
- `Database/script.sql` (legacy schema)

## 1) Which SQL file to use

Use `Database/supabase_setup.sql` for Supabase.

Why:
- It links `public.users.user_id` to `auth.users(id)`.
- It enables Row Level Security (RLS).
- It adds policies so each user only accesses their own data.
- It includes secure helper triggers/functions for user sync.

`Database/script.sql` is legacy and stores `password_hash` + `password_salt` in app tables. This is **not needed** when using Supabase Auth.

## 2) Setup order (important)

1. Create a new Supabase project.
auracheck, pwd-auracheck@2709
2. In Supabase Dashboard, go to **SQL Editor**.
3. Run the full `Database/supabase_setup.sql` script.
4. Confirm tables are created:
   - `public.users`
   - `public.profile`
   - `public.daily_inputs`
5. Confirm RLS is enabled on all 3 tables.
6. Confirm triggers exist:
   - `on_auth_user_created`
   - `on_auth_user_updated`

## 3) App environment setup

Add to your project `.env`:

```env
SUPABASE_URL=<your-project-url>
SUPABASE_ANON_KEY=<your-anon-key>
# Only for trusted backend/admin jobs (never frontend/client)
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
```

Security rules:
- Use `SUPABASE_ANON_KEY` for user-facing app calls.
- Use `SUPABASE_SERVICE_ROLE_KEY` only in trusted server contexts.
- Never commit `.env`.
- Never log keys in app output.

## 4) Supabase Auth hardening (must do)

In **Authentication** settings:
- Enable email confirmation.
- Disable anonymous sign-ins unless required.
- Enforce stronger password policy (minimum length + complexity).
- Configure allowed redirect URLs to only your real domains/environments.
- Remove wildcard redirects in production.

## 5) Database hardening (must do)

The SQL already enables RLS + policies. Keep these practices:

- Do not grant broad privileges to `anon`.
- Keep all user tables in `public` behind RLS.
- Use `authenticated` role in policies (already done).
- Keep admin access policy based on JWT app metadata role (`admin`) as in script.
- Do not bypass RLS in application code.

Recommended additional SQL checks:

```sql
-- RLS status
select tablename, rowsecurity
from pg_tables
where schemaname = 'public'
  and tablename in ('users','profile','daily_inputs');

-- Existing policies
select schemaname, tablename, policyname, roles, cmd
from pg_policies
where schemaname = 'public'
  and tablename in ('users','profile','daily_inputs')
order by tablename, policyname;
```

## 6) Data protection measures

- Minimize stored personal data (collect only fields you need).
- Keep sensitive data in `jsonb` structured objects only when necessary.
- Avoid storing secret tokens in table columns.
- Use `on delete cascade` relationships carefully (already defined) and back up data.
- Turn on PITR/backups (Supabase project settings) for recovery.
- Set retention and deletion policy for old `daily_inputs` records.

## 7) Connection security

- Supabase API connections are HTTPS by default; keep it that way.
- If using direct Postgres connections for admin scripts, enforce `sslmode=require`.
- Store DB connection strings in secret managers/CI secrets, not source code.
- Rotate keys on schedule and immediately after any suspected leak.

## 8) Verification checklist (after setup)

- [ ] New user signup creates row in `public.users` via `on_auth_user_created` trigger.
- [ ] Email verification updates `public.users.is_verified`.
- [ ] Authenticated user can only read/write their own row(s).
- [ ] Another authenticated user cannot access someone else’s data.
- [ ] `daily_inputs` cannot be updated/deleted by authenticated users (as intended by script).
- [ ] Admin role in JWT can read all rows for monitoring.

## 9) Important notes for current codebase

Current `app.py` still includes SQLite local DB paths/functions.
If you want full Supabase-only mode, migrate app data operations from SQLite to Supabase client calls and remove local password handling paths.

---

If you need, next step I can generate a migration plan to move your current `app.py` CRUD functions (`create_user`, `authenticate_user`, `upsert_profile`, `upsert_daily_input`) from SQLite to Supabase safely.
