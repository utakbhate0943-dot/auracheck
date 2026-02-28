# AuraCheck Internal Setup Guide

This document is for the internal engineering team.

## 1) Prerequisites

- Python 3.10+ (recommended: 3.11)
- Access to Supabase project (URL + anon key)
- OpenAI API key
- Git access to this repository

## 2) Local environment setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Required environment variables

Create `.env` at root:

```env
SUPABASE_URL=<supabase-url>
SUPABASE_ANON_KEY=<supabase-anon-key>
OPENAI_API_KEY=<openai-key>
OPENAI_MODEL=gpt-4.1-mini
```

## 4) Supabase schema (required)

```sql
create table if not exists profiles (
  user_id uuid primary key,
  phone text,
  full_name text,
  age int,
  gender text,
  course text,
  goals text,
  created_at timestamp with time zone default now(),
  updated_at timestamp with time zone default now()
);

create table if not exists daily_logs (
  id bigint generated always as identity primary key,
  user_id uuid,
  log_date date default current_date,
  predicted_stress_level int,
  predicted_mental_health_pct numeric,
  mood_score int,
  sleep_hours numeric,
  remark text,
  recommendation text,
  created_at timestamp with time zone default now()
);

create index if not exists daily_logs_user_id_idx on daily_logs(user_id);
create index if not exists daily_logs_user_date_idx on daily_logs(user_id, log_date);
```

Expected source table for model dataset sync:

- `students_mental_health` (used by `training_module/fetch_supabase_data.py`)

## 5) Dataset column expectations

The model pipeline expects at minimum:

- `Stress_Level` (required for supervised training)

Common feature columns used in UI/model input:

- `Age`, `Gender`, `Course`, `CGPA`
- `Sleep_Quality`, `Physical_Activity`, `Diet_Quality`
- `Social_Support`, `Financial_Stress`
- `Extracurricular_Involvement`, `Semester_Credit_Load`, `Residence_Type`
- Optional numeric context: `Depression_Score`, `Anxiety_Score`

If columns are missing, training now fails early with explicit validation error messages.

## 6) Run locally

```bash
streamlit run app.py
```

## 7) Troubleshooting

- OpenAI 429/insufficient quota:
  - Verify billing/plan, or rely on built-in fallback recommendations.
- Supabase auth/link not working:
  - Validate `SUPABASE_URL`, `SUPABASE_ANON_KEY`, and allowed redirect URLs.
- Model setup failed due to schema:
  - Ensure `students_mental_health` table has required columns.
- Empty dashboard on Page 2:
  - Ensure user logged in and `daily_logs` rows exist for that user.

## 8) Internal release checklist

- [ ] `pip install -r requirements.txt` passes on clean venv
- [ ] App launches (`streamlit run app.py`)
- [ ] Page 1: prediction + gauges + recommendations render
- [ ] Page 2: profile save + daily check-in + charts render
- [ ] Supabase write/read path verified in staging
- [ ] OpenAI fallback path tested (without API key or quota)

## 9) Ownership

- App/UI orchestration: `app.py`
- Training pipelines: `training_module/*`
- Data fetch from Supabase: `training_module/fetch_supabase_data.py`