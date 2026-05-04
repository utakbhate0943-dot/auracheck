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


## System Architecture Diagrams

### End-to-End System Architecture
![AuraCheck Architecture](Dataset/figures/Auracheck%20Architecture.png)

### Authentication Layer
![Authentication Layer](Dataset/figures/Auracheck%20Authetication%20Layer.png)

### ML Model Integration
![ML Model Integration](Dataset/figures/Auracheck%20ML%20Integration.png)

### Supabase Database Tables
![Supabase Database Tables](Dataset/figures/Auracheck%20Supabase.png)

### Tech Stack Overview
![Tech Stack](Dataset/figures/Auracheck%20Techstack.png)

## Application Flow

1. **User Input & UI:**
	- Users interact with a Streamlit UI (`app.py`) to submit wellbeing survey answers.

2. **Authentication & User Management:**
	- Users sign up/log in via Supabase (with hashed/salted passwords).
	- Password reset is handled via Supabase email functionality.
	- User credentials and profiles are stored in Supabase.

3. **Data Storage:**
	- User responses are saved both locally (as JSON) and in Supabase (`daily_inputs` table).
	- If Supabase is unavailable, local JSON is used as a fallback.

4. **ML Integration:**
	- User answers are passed to a trained Random Forest model (see `scripts/integrated_model_inference.py`).
	- The model predicts burnout class and returns results for display.

5. **Results & History:**
	- Results are shown to the user immediately.
	- If logged in, results are saved to their Supabase history.
	- Users can view their historical progress and trends.

6. **Progress Tracking:**
	- Users can see their daily history and progress charts.
	- Data syncs between local and Supabase when possible.

## Supabase & ML Integration

- **Supabase:** Handles authentication, user management, password reset, and persistent storage of user responses and profiles.
- **Random Forest Model:** Receives normalized user input, predicts burnout class, and returns results for display and storage.

---

## Presentation Slides (Markdown)

### Slide 1: Overall App Flow

- User accesses AuraCheck via web UI.
- Signs up/logs in (Supabase authentication).
- Completes daily wellbeing questionnaire.
- Receives burnout risk prediction.
- Results and history are saved and visualized.

### Slide 2: Supabase Integration

- User credentials securely stored (hash + salt).
- Daily responses and profile data saved in Supabase tables.
- Password reset via Supabase email.
- Syncs local data to Supabase when online.
- Enables user history and progress tracking.

### Slide 3: ML Integration (Random Forest)

- User answers are normalized and passed to a trained Random Forest model.
- Model predicts burnout class (low/mid/high).
- Model artifacts and inference logic in `ml_randomforest/` and `scripts/integrated_model_inference.py`.
- Results shown instantly and saved for history.

### Slide 4: Future Scope & Details

- Expand ML models (e.g., deep learning, ensemble).
- Add more wellbeing metrics and recommendations.
- Enhance visualizations and user feedback.
- Integrate with more data sources (wearables, etc.).
- Improve personalization and notifications.

## Repository Highlights

- app.py: main application flow, auth, questionnaire, charting, and persistence.
- baseline/scripts/production_pruned_multinomial_baseline.py: baseline training and inference artifact generation.
- baseline/scripts/create_final_process_baseline_report.py: candidate comparison and final report generation.
- EDA/Edgar/eda_augmented_deep.py: deep EDA workflow and output generation.
- Unsupervised/README.md: clustering experiments for burnout recovery, focused on K-Means baseline artifacts and KMeans benchmark comparisons.
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
- Unsupervised baseline outputs are generated in Unsupervised/outputs/baseline_kmeans.
- Unsupervised benchmark outputs are generated in Unsupervised/outputs/kmeans_benchmark.

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
