# AuraCheck Presentation Deck (Content)

Use this file as speaker notes / slide content.

## Slide 1 — Title

**AuraCheck**

AI-assisted student mental wellness assessment and progress tracking platform.

## Slide 2 — Problem Statement

- Students face increasing stress, anxiety, and inconsistent support access.
- Existing solutions are often fragmented (assessment, support, tracking in separate tools).
- Teams need a lightweight, private, actionable wellness workflow.

## Slide 3 — Solution Overview

AuraCheck combines:

- Smart questionnaire + model-based stress analysis
- Mental health percentage estimate with visual gauges
- Behavior grouping via KMeans
- Recommendation layer via OpenAI (with fallback)
- Profile + daily tracker dashboard for continuous monitoring

## Slide 4 — User Journey

1. User opens Page 1
2. Completes questionnaire
3. Gets stress + mental health analysis + behavior remark
4. Receives AI recommendations
5. Signs up/login via email magic link
6. Uses Page 2 for daily check-ins and trend dashboard

## Slide 5 — Tech Stack

### Frontend / App Layer

- Streamlit
- Plotly (gauges)

### Data + ML

- Pandas, NumPy
- scikit-learn
  - Logistic Regression
  - Gradient Boosting
  - KMeans

### Backend Services

- Supabase
  - Auth (magic link/OTP)
  - Postgres tables (`profiles`, `daily_logs`, `students_mental_health`)

### AI Layer

- OpenAI Responses API
- Configurable model via env (`OPENAI_MODEL`)

## Slide 6 — Architecture (Logical)

- Streamlit UI gathers inputs
- Dataset fetched/synced from Supabase to local CSV
- Model training bundle initialized at startup
- Prediction + clustering run on questionnaire submission
- Results rendered as cards + gauges + remarks
- Recommendations generated through OpenAI or fallback template
- User logs and profile persisted in Supabase

## Slide 7 — Data Model

### `profiles`

- `user_id`, `full_name`, `age`, `gender`, `course`, `goals`, timestamps

### `daily_logs`

- `user_id`, `log_date`
- `predicted_stress_level`, `predicted_mental_health_pct`
- `mood_score`, `sleep_hours`
- `remark`, `recommendation`

## Slide 8 — Current Capabilities

- Multi-model stress prediction
- Mental health gauge visualization
- Behavior-group remarking
- Personalized recommendation generation
- Daily tracker dashboard with trend lines
- Explicit column/schema validation for safer operations

## Slide 9 — Risks & Mitigations

- API quota limits (OpenAI)
  - Mitigation: safe fallback recommendation path
- Dataset schema drift
  - Mitigation: required-column validation with explicit errors
- Auth and data access misconfiguration
  - Mitigation: internal setup and release checklist

## Slide 10 — Weekly 8-Week Plan

### Week 1 — Foundation & Environment

- Finalize repo structure, env templates, baseline docs
- Validate Supabase schema in dev/staging
- Set coding standards + issue templates

### Week 2 — Data & Pipeline Reliability

- Stabilize Supabase-to-CSV sync
- Add schema validation checks for training input
- Add sanity checks for missing/null-heavy columns

### Week 3 — Model Improvements (Supervised)

- Baseline metrics for Logistic + Gradient Boosting
- Hyperparameter tuning pass
- Add model comparison summary output

### Week 4 — Unsupervised Insights

- Calibrate KMeans cluster definitions
- Add interpretable cluster behavior descriptions
- Validate cluster consistency against historical data

### Week 5 — Recommendation Intelligence

- Improve OpenAI prompt quality and response formatting
- Introduce recommendation categories (immediate / weekly / support)
- Add retry/backoff guidance for API failures

### Week 6 — UX & Dashboard Refinement

- Improve compact layout consistency
- Add stronger visual hierarchy for key outputs
- Improve dashboard readability and trend interpretation

### Week 7 — Security, QA, and Hardening

- End-to-end testing for auth/data/model flow
- Add RLS policy guidance and role checks
- Performance checks for startup and prediction latency

### Week 8 — Release Readiness & Handoff

- Final bug bash + UAT with stakeholder feedback
- Prepare rollout notes and operational runbook
- Team enablement: demo, docs walkthrough, ownership matrix

## Slide 11 — Success Metrics

- Prediction response time
- Recommendation generation success rate
- Daily/weekly active usage
- Check-in retention (7-day, 30-day)
- % users completing Page 1 → Page 2 journey

## Slide 12 — Next Phase

- Role-based portals (student/counselor/admin)
- Longitudinal risk trend detection
- Intervention alerts and escalation workflows