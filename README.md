# AuraCheck

AuraCheck is a Streamlit application for student mental wellness screening and daily progress tracking.

## What it does

- Collects questionnaire inputs and predicts **Stress Level** using two supervised models:
	- Logistic Regression
	- Gradient Boosting
- Estimates **Mental Health %** and visualizes results using gauge charts
- Uses **KMeans clustering** to provide behavior-group remarks
- Generates practical recommendations using **OpenAI**
- Supports email magic-link login with **Supabase Auth**
- Provides profile + daily tracker dashboard on Page 2

## Quick start

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

### 2) Configure environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_ANON_KEY=your_supabase_anon_key
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

### 3) Create Supabase tables

Run this SQL in Supabase SQL Editor:

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

### 4) Run the app

```bash
streamlit run app.py
```

## App pages

- **Page 1 — Assessment + Chat**
	- Questionnaire
	- Stress / Mental Health analysis cards + gauges
	- KMeans behavior remark
	- OpenAI recommendations
	- Signup/Login prompt
- **Page 2 — Profile + Dashboard**
	- Profile management
	- Daily check-in
	- Trend dashboard for stress, health %, mood, and sleep

## Notes

- If OpenAI quota is unavailable, the app returns safe fallback recommendations.
- For internal developer setup and deployment checklist, see `INTERNAL_SETUP.md`.
- For architecture/tech-stack presentation and roadmap, see `PRESENTATION.md`.
