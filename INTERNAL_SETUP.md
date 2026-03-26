# AuraCheck Internal Setup Guide

This guide is for internal contributors and collaborators.

## 1) Prerequisites

- Python 3.10+ (recommended: 3.11)
- Git access to this repository
- Optional: Supabase project access
- Optional: OpenAI API key

## 2) Local Environment Setup

Use a virtual environment:

python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

## 3) Environment Variables

Create .env in project root.

Minimum for local-only mode:

- No required variables.

Optional variables:

- OPENAI_API_KEY
- OPENAI_MODEL=gpt-4.1-mini
- SUPABASE_URL
- SUPABASE_ANON_KEY
- SUPABASE_SERVICE_ROLE_KEY
- ADMIN_EMAIL

Security rules:

- Never commit .env.
- Never share keys in docs, commits, chat, or screenshots.

## 4) Runtime Modes

AuraCheck currently runs SQLite-first and can mirror writes to Supabase.

- Local mode:
  - Uses Database/auracheck.db created automatically.
  - Works without Supabase credentials.
- Hybrid mode:
  - Writes locally first.
  - Attempts Supabase upsert if configured.
  - Local save is not blocked by Supabase DNS/network failures.

## 5) Database Setup (Supabase Optional)

Recommended SQL sequence:

1. Run Database/supabase_setup.sql for secure baseline schema and RLS.
2. If using current app-managed SQLite auth mirror behavior, run Database/supabase_fix_for_app_sqlite_auth.sql.

Notes:

- Database/script.sql is legacy and keeps password hash/salt columns; use only if intentionally running that legacy model.
- Preferred reference guide: Database/SUPABASE_SETUP_GUIDE.md.

## 6) Run Application

streamlit run app.py

Main behaviors to verify:

- Questionnaire flow and scoring.
- Daily submission save in SQLite.
- Profile save and progress charts.
- Admin view access only for configured admin email.

## 7) Baseline Model Scripts

Primary script:

- baseline/scripts/production_pruned_multinomial_baseline.py

Purpose:

- Trains balanced multinomial logistic baseline.
- Saves model, metadata, metrics, and example prediction artifacts.

Run:

python baseline/scripts/production_pruned_multinomial_baseline.py

Outputs:

- baseline/outputs/final_baseline_model/production_pruned_multinomial_model.joblib
- baseline/outputs/final_baseline_model/production_pruned_multinomial_metadata.json
- baseline/outputs/final_baseline_model/production_pruned_multinomial_metrics.csv

## 8) Baseline Comparison Report

Script:

- baseline/scripts/create_final_process_baseline_report.py

Run:

python baseline/scripts/create_final_process_baseline_report.py

Outputs include:

- final_process_baseline_comparison_report.pdf
- final_process_baseline_comparison_report_summary.txt
- confusion matrix and sensitivity/specificity CSV files

## 9) EDA Workflow

Primary EDA script:

- EDA/Edgar/eda_augmented_deep.py

Run:

python EDA/Edgar/eda_augmented_deep.py

Generated outputs:

- EDA/Edgar/outputs/augmented_deep/analysis_summary.json
- EDA/Edgar/outputs/augmented_deep/summary_text.txt
- EDA/Edgar/outputs/augmented_deep/figures/*

Optional table export script:

- EDA/Edgar/export_selected_tables_to_pdf.py

## 10) Troubleshooting

- Supabase not syncing:
  - Confirm SUPABASE_URL and key values.
  - Verify project is reachable and DNS is resolving.
  - Local SQLite saves should still succeed.
- OpenAI unavailable or quota errors:
  - Ensure OPENAI_API_KEY is valid.
  - Verify fallback recommendation behavior in app.
- Import errors in EDA or baseline scripts:
  - Reinstall requirements in a clean environment.
- Empty user progress section:
  - Confirm user login and at least one successful daily submission.

## 11) Internal Release Checklist

- [ ] Fresh environment installs from requirements.txt successfully.
- [ ] streamlit run app.py launches.
- [ ] Local signup/login/profile flow works.
- [ ] Daily input writes to SQLite and appears in progress charts.
- [ ] Baseline scripts run and regenerate expected artifacts.
- [ ] EDA script runs and generates output summaries/figures.
- [ ] Supabase mirror path tested if credentials are configured.

## 12) Code Ownership Map

- App orchestration, UI, persistence: app.py
- Baseline training and report generation: baseline/scripts/
- SQL setup and schema hardening: Database/
- Exploratory analytics: EDA/