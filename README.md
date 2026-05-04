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
