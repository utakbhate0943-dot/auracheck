# AuraCheck

AuraCheck is a student wellbeing check-in application built with Streamlit. It combines a guided questionnaire, lightweight stress scoring, daily progress tracking, and baseline model research artifacts in one repository.

## Public Project Summary

This repository includes four main workstreams:

- Application: a Streamlit app for daily wellbeing check-ins and progress views.
- Baseline modeling: scripts and outputs for a production baseline multinomial classifier.
- EDA: exploratory analysis scripts and generated analysis artifacts.
- Database: SQL scripts for local and Supabase-backed persistence options.

## Key Features

- Guided multi-step wellbeing questionnaire.
- Daily score outputs (stress and wellbeing indicators) with visual gauges.
- Per-user daily history and trend charts.
- Local SQLite persistence by default, with optional Supabase mirror sync.
- Baseline model training and reproducible output artifacts.
- EDA pipeline producing summary tables, diagnostics, and figures.

## High-Level Architecture

- Frontend and app logic: Streamlit in app.py.
- Local persistence: SQLite database stored under Database/auracheck.db.
- Optional remote sync: Supabase client upserts to users, profile, and daily_inputs tables.
- AI recommendations: OpenAI client is supported when API key is configured.
- Baseline model artifacts: saved under baseline/outputs/final_baseline_model.

## Repository Highlights

- app.py: main application flow, auth, questionnaire, charting, and persistence.
- baseline/scripts/production_pruned_multinomial_baseline.py: baseline training and inference artifact generation.
- baseline/scripts/create_final_process_baseline_report.py: candidate comparison and final report generation.
- EDA/Edgar/eda_augmented_deep.py: deep EDA workflow and output generation.
- Unsupervised/README.md: clustering experiments for burnout recovery, focused on K-Means and selector-ranked comparisons.
- Database/supabase_setup.sql: recommended Supabase schema and security policies.
- Database/supabase_fix_for_app_sqlite_auth.sql: compatibility patch for app-managed auth mirroring.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies.
3. Add environment variables.
4. Run the Streamlit app.

Example commands:

pip install -r requirements.txt
streamlit run app.py

## Environment Variables

Create a local .env file at repository root.

Required for app startup:

- None for local-only mode.

Optional:

- OPENAI_API_KEY
- OPENAI_MODEL (default: gpt-4.1-mini)
- SUPABASE_URL
- SUPABASE_ANON_KEY
- SUPABASE_SERVICE_ROLE_KEY
- ADMIN_EMAIL

Security notes:

- Do not commit .env.
- Do not share keys or service-role secrets in docs, screenshots, or logs.

## Database Modes

- Local mode (default): uses SQLite only.
- Hybrid mode: SQLite plus optional Supabase mirror sync.

For Supabase schema setup, see Database/SUPABASE_SETUP_GUIDE.md and SQL files in Database/.

## Baseline and EDA Artifacts

- Baseline metrics and model metadata are generated in baseline/outputs/final_baseline_model.
- EDA outputs (summary text, CSV diagnostics, figures) are generated in EDA/Edgar/outputs/augmented_deep.
- Unsupervised experiment outputs are generated in Unsupervised/outputs/experiments.

## Responsible Use

AuraCheck is for educational and informational purposes. It is not a medical diagnosis tool and should not replace licensed clinical care.

## Team Setup

For internal setup, runbooks, and troubleshooting, see INTERNAL_SETUP.md.

## Pre-Commit Secret Scanning

This repository uses pre-commit hooks to block accidental secret commits.

Setup once per clone:

1. Install pre-commit:
	pip install pre-commit
2. Install hooks:
	pre-commit install

Optional manual scan of all tracked files:

pre-commit run --all-files

Notes:

- .env remains ignored and is excluded from commit-time scanning because it is local-only.
- Secret-like values in staged files will fail the commit until removed or replaced.
