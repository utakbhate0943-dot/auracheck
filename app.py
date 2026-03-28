"""AuraCheck Streamlit application.

This module intentionally keeps UI rendering, authentication, local persistence,
and optional Supabase sync together to simplify project delivery.
"""

import os
import json
import sqlite3
import uuid
import secrets
import hashlib
import hmac
from datetime import date, datetime
from typing import Optional
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client

load_dotenv(override=True)

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "utakbhate0943@sdsu.com").strip().lower()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(APP_DIR, "Database")
DB_PATH = os.path.join(DB_DIR, "auracheck.db")
USER_RESPONSES_JSON_PATH = os.path.join(APP_DIR, "Dataset", "user_responses.json")

REQUIRED_FIELDS = [
    "Age", "Course", "Gender", "CGPA", "Sleep_Quality",
    "Physical_Activity", "Diet_Quality", "Social_Support",
    "Relationship", "Substance_Use", "Counseling",
    "Family_History", "Chronic_Illness", "Financial_Stress",
    "Extracurricular", "Semester", "Residence_Type",
]

POSITIVE_THOUGHTS = [
    "🌟 You are capable of overcoming challenges",
    "💚 Your mental health matters and deserves attention",
    "🌈 Every day is a fresh opportunity for growth",
    "💫 You have strength within you",
    "🌸 Self-care is not selfish, it's essential",
    "⭐ Progress over perfection always",
    "🎯 Your feelings are valid and important",
    "🌊 Challenges help you grow stronger",
    "💡 You deserve to be happy and healthy",
    "🦋 Transformation starts with self-compassion",
]

# These fields are treated as stable baseline attributes after first response.
STATIC_USER_FIELDS = {
    "Age": "survey_age",
    "Course": "survey_course",
    "Gender": "survey_gender",
}


def normalize_supabase_url(url_value: str) -> str:
    """Normalize Supabase URL so env variants do not break client setup."""
    normalized = (url_value or "").strip().strip('"').strip("'").rstrip("/")
    if normalized and not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    return normalized


SUPABASE_URL = normalize_supabase_url(os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or "").strip()

st.set_page_config(page_title="AuraCheck", page_icon="💜", layout="wide")

# Custom CSS - Light purple background with single card design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Light purple background */
    .stApp {
        background: linear-gradient(135deg, #E3D7F2 0%, #ECE0F7 50%, #DDCFEC 100%);
        min-height: 100vh;
        padding: 6px 15px;
    }

    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Main container card - large single card */
    .main {
        background: linear-gradient(135deg, rgba(248, 244, 253, 0.95) 0%, rgba(244, 238, 251, 0.95) 50%, rgba(239, 232, 248, 0.95) 100%);
        border-radius: 40px;
        padding: 0;
        box-shadow: 0 35px 100px rgba(124, 91, 166, 0.3), 0 15px 50px rgba(155, 127, 181, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.7);
        margin: 25px auto;
        max-width: 750px;
        width: 100%;
        border: 5px solid #705291;
        overflow: hidden;
    }
    
    /* Header section - logo and title */
    .header-section {
        background: transparent;
        padding: 65px 50px 40px 50px;
        text-align: center;
    }
    
    /* Questions section */
    .questions-section {
        padding: 45px 50px;
        background: transparent;
    }
    
    /* Results section */
    .results-section {
        padding: 45px 50px;
        background: transparent;
    }
    
    /* Analyze button section */
    .analyze-section {
        background: transparent;
        padding: 35px 50px;
        text-align: center;
    }
    
    /* Footer section - Login/Signup */
    .footer-section {
        background: transparent;
        padding: 40px 50px 50px 50px;
    }
    
    /* Button styling for questions */
    .stButton > button {
        width: 100%;
        border-radius: 18px;
        padding: 18px 28px;
        font-size: 17px;
        font-weight: 600;
        border: 3px solid #355DCB !important;
        background-color: #FFFFFF !important;
        color: #1F3F9F !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 14px;
        box-shadow: 0 6px 16px rgba(53, 93, 203, 0.25), 0 2px 8px rgba(0, 0, 0, 0.12) !important;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.4px;
        text-transform: none;
        line-height: 1.5;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #355DCB 0%, #234AAE 100%) !important;
        color: #FFFFFF !important;
        transform: translateY(-4px);
        border-color: #355DCB !important;
        box-shadow: 0 16px 45px rgba(53, 93, 203, 0.45), 0 8px 20px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stButton > button:active {
        transform: translateY(-2px);
    }
    
    /* Special styling for Analyze button */
    .analyze-btn {
        background: linear-gradient(135deg, #9B7FB5 0%, #8B6FA5 100%) !important;
        color: #FFFFFF !important;
        border: 3px solid #9B7FB5 !important;
        font-size: 18px !important;
        padding: 20px 35px !important;
    }
    
    .analyze-btn:hover {
        background: linear-gradient(135deg, #8B6FA5 0%, #7B5F95 100%) !important;
    }
    
    /* Login/Signup buttons footer styling */
    .login-signup-container {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin-top: 25px;
    }
    
    .login-button {
        border: 3px solid #8B7BA8 !important;
        color: #8B7BA8 !important;
        background: #FFFFFF !important;
        flex: 1;
        font-weight: 700 !important;
    }
    
    .login-button:hover {
        background: linear-gradient(135deg, #8B7BA8 0%, #7B6B98 100%) !important;
        color: #FFFFFF !important;
    }
    
    .signup-button {
        border: 3px solid #9B7FB5 !important;
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #9B7FB5 0%, #8B6FA5 100%) !important;
        flex: 1;
        font-weight: 700 !important;
    }
    
    .signup-button:hover {
        background: linear-gradient(135deg, #8B6FA5 0%, #7B5F95 100%) !important;
    }
    
    /* Typography */
    h1 {
        text-align: center;
        color: #3F2456;
        font-size: 48px;
        margin: 15px 0 8px 0;
        font-weight: 800;
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.8px;
        line-height: 1.1;
    }
    
    h2 {
        text-align: center;
        color: #5A3D79;
        font-size: 20px;
        font-weight: 500;
        margin: 0 0 30px 0;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.3px;
    }
    
    h3 {
        color: #3F2456;
        font-family: 'Poppins', sans-serif;
        font-size: 28px;
        font-weight: 700;
        margin-top: 0;
        margin-bottom: 20px;
        letter-spacing: -0.3px;
    }
    
    h4 {
        color: #3D2C55;
        font-family: 'Inter', sans-serif;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Question text */
    .question-text {
        text-align: center;
        color: #2F2142;
        font-size: 24px;
        margin: 30px 0 35px 0;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.15px;
        line-height: 1.4;
    }
    
    /* Logo styling */
    .logo {
        text-align: center;
        margin: 0 0 20px 0;
        animation: pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    .logo img {
        animation: pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.06); }
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #9B7FB5 0%, #8B6FA5 100%) !important;
        border-radius: 12px;
        height: 8px;
    }
    
    .stProgress {
        background-color: rgba(155, 127, 181, 0.15) !important;
        border-radius: 12px;
        margin-bottom: 22px;
        height: 8px;
    }
    
    /* Progress text */
    .progress-text {
        text-align: center;
        color: #5A3D79;
        font-size: 13px;
        font-weight: 600;
        margin-top: 12px;
        letter-spacing: 0.4px;
    }
    
    /* Dividers - hidden */
    hr {
        border: none;
        border-top: none;
        margin: 0;
        display: none !important;
    }
    
    /* Text styling */
    p, span, label {
        color: #352549;
        font-weight: 500;
        line-height: 1.6;
    }

    .stTextInput label {
        color: #2F2142 !important;
        font-weight: 600 !important;
    }

    .stTextInput input {
        color: #221532 !important;
        background-color: #FFFFFF !important;
        border: 2px solid #6C4B92 !important;
    }

    .stTextInput input::placeholder {
        color: #6A5A82 !important;
    }
    
    /* Metric containers */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #F5F0FF 0%, #FAFBFF 100%);
        border-radius: 18px;
        padding: 24px;
        border: 2px solid #D4C5E2;
        box-shadow: 0 4px 12px rgba(155, 127, 181, 0.15);
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        border: 2px solid #D4C5E2 !important;
        border-radius: 14px !important;
        background: linear-gradient(135deg, #FAFBFF 0%, #F5F8FF 100%) !important;
    }
    
    /* Animation */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .stButton {
        animation: slideUp 0.5s ease-out;
    }
    
    /* Card styling */
    .card-style {
        background: linear-gradient(135deg, #FFFFFF 0%, #FAFBFF 100%);
        border-radius: 16px;
        padding: 24px;
        border: 2px solid #E8DFF5;
        box-shadow: 0 4px 15px rgba(155, 127, 181, 0.1), 0 2px 8px rgba(0, 0, 0, 0.05);
        margin-bottom: 18px;
    }
    
    /* Footer text */
    .footer-text {
        text-align: center;
        color: #7A6B8F;
        font-size: 15px;
        margin-bottom: 25px;
        font-weight: 500;
    }
    
    /* Shiny purple placeholder - big background container */
    .content-placeholder {
        background: linear-gradient(135deg, rgba(229, 216, 241, 0.82) 0%, rgba(222, 208, 236, 0.8) 50%, rgba(216, 199, 232, 0.84) 100%);
        border-radius: 32px;
        padding: 0;
        position: relative;
        overflow: hidden;
        margin: 20px;
    }
    
    .content-placeholder::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(
            45deg,
            transparent 25%,
            rgba(255, 255, 255, 0.12) 50%,
            transparent 75%
        );
        animation: shimmer 3.5s infinite;
        pointer-events: none;
        z-index: 1;
    }
    
    @keyframes shimmer {
        0% {
            transform: translateX(-150%);
        }
        100% {
            transform: translateX(150%);
        }
    }
    
    .content-placeholder > * {
        position: relative;
        z-index: 2;
    }
    
    /* Adjust section styles to work inside placeholder */
    .header-section {
        background: transparent;
        padding: 60px 50px 35px 50px;
        text-align: center;
    }
    
    .questions-section {
        padding: 40px 50px;
        background: transparent;
    }
    
    .results-section {
        padding: 40px 50px;
        background: transparent;
    }
    
    .analyze-section {
        background: transparent;
        padding: 30px 50px;
        text-align: center;
    }
    
    .footer-section {
        background: transparent;
        padding: 35px 50px 45px 50px;
        border-radius: 0 0 30px 30px;
    }
    
    /* 3-Column Layout Styles */
    .left-section {
        background: transparent;
        padding: 0;
        border-radius: 20px;
        text-align: center;
        height: auto;
        min-height: 0;
        display: block;
    }
    
    .middle-section {
        background: transparent;
        padding: 40px 32px 44px 32px;
        border-radius: 24px;
        height: auto;
        min-height: 0;
        min-width: 760px;
        max-width: 760px;
        margin: 0 auto;
        display: block;
       
    }

    .middle-panel {
        background: #ffffff;
        border: none;
        border-radius: 0;
        padding: 0;
        box-shadow: none;
    }
    
    .right-section {
        background: transparent;
        padding: 0;
        border-radius: 20px;
        display: block;
        height: auto;
        min-height: 0;
        min-width: 300px;
        max-width: 300px;
        margin: 0 auto;
    }
    
    /* Good Thoughts Section */
    .good-thoughts-header {
        color: #2D1A42;
        font-size: 22px;
        font-weight: 700;
        text-align: center;
        margin: 35px 0 20px 0;
        letter-spacing: 0.3px;
        font-family: 'Poppins', sans-serif;
    }
    
    .good-thoughts-container {
        background: transparent;
        border-radius: 14px;
        margin: 0;
        padding: 8px 0;
    }
    
    .thought-item {
        color: #5B4B6F;
        font-size: 25px;
        font-weight: 700;
        text-align: center;
        padding: 16px 16px;
        margin: 30px 0;
        background: rgba(155, 127, 181, 0.15);
        border-radius: 10px;
        border: 1px solid rgba(155, 127, 181, 0.15);
        line-height: 1.5;
        transition: all 0.6s ease;
    }
    
    .thought-item:hover {
        background: linear-gradient(135deg, rgba(155, 127, 181, 0.15) 0%, rgba(200, 170, 220, 0.15) 100%);
        border-color: rgba(155, 127, 181, 0.3);
        transform: translateY(-2px);
    }

    .thought-animated-box {
        min-height: 120px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .left-section h1 {
        font-size: 56px;
        margin: 18px 0 10px 0;
    }

    .left-section h2 {
        font-size: 32px;
        font-weight: 600;
        margin: 0 0 24px 0;
    }
    
    /* Auth Section - Right Column */
    .auth-header {
        color: #3F2456;
        font-size: 18px;
        font-weight: 700;
        text-align: center;
        margin: 0 0 15px 0;
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.3px;
    }
    
    .auth-subtext {
        color: #4D3468;
        font-size: 12px;
        text-align: center;
        line-height: 1.5;
        font-weight: 600;
        letter-spacing: 0.2px;
    }
    
    /* Responsive adjustments for smaller screens */
    @media (max-width: 1200px) {
        .left-section h1 {
            font-size: 44px;
        }
        
        .left-section h2 {
            font-size: 24px;
        }
        
        .thought-item {
            font-size: 17px;
            padding: 14px 12px;
        }

        .good-thoughts-header {
            font-size: 20px;
        }

        .middle-section {
            min-width: auto;
            max-width: 100%;
            height: auto;
            min-height: 0;
        }

        .right-section {
            min-width: auto;
            max-width: 100%;
            height: auto;
            min-height: 0;
        }

        .left-section {
            height: auto;
            min-height: 0;
        }
    }
    
</style>
""", unsafe_allow_html=True)


# =========================================
# Prediction and questionnaire helper methods
# =========================================
def get_static_models():
    """Return static/mock model data for demonstration."""
    return {
        "logistic_regression": {
            "feature_cols": ["Age", "Course", "Gender", "CGPA", "Sleep_Quality", 
                           "Physical_Activity", "Diet_Quality", "Social_Support", 
                           "Relationship", "Substance_Use", "Counseling", 
                           "Family_History", "Chronic_Illness", "Financial_Stress", 
                           "Extracurricular", "Semester", "Residence_Type"],
            "accuracy": 0.85,
            "f1_score": 0.82,
        },
        "gradient_boosting": {
            "feature_cols": ["Age", "Course", "Gender", "CGPA", "Sleep_Quality", 
                           "Physical_Activity", "Diet_Quality", "Social_Support", 
                           "Relationship", "Substance_Use", "Counseling", 
                           "Family_History", "Chronic_Illness", "Financial_Stress", 
                           "Extracurricular", "Semester", "Residence_Type"],
            "accuracy": 0.88,
            "f1_score": 0.85,
        },
        "kmeans": {
            "feature_cols": ["Sleep_Quality", "Physical_Activity", "CGPA", 
                           "Social_Support", "Counseling", "Financial_Stress"],
            "silhouette_score": 0.42,
        }
    }


def predict_stress_level(answers: dict, model_type: str = "logistic") -> int:
    """Generate stress prediction based on input answers."""
    stress_factors = 0
    
    if "Sleep_Quality" in answers:
        try:
            sleep_val = float(answers["Sleep_Quality"]) if isinstance(answers["Sleep_Quality"], (int, float)) else 3
            stress_factors += (5 - sleep_val)
        except:
            stress_factors += 2
    
    if "Financial_Stress" in answers:
        try:
            fin_val = float(answers["Financial_Stress"]) if isinstance(answers["Financial_Stress"], (int, float)) else 5
            stress_factors += fin_val / 2
        except:
            stress_factors += 2.5
    
    if "Physical_Activity" in answers:
        try:
            activity_val = float(answers["Physical_Activity"]) if isinstance(answers["Physical_Activity"], (int, float)) else 3
            stress_factors += (5 - activity_val) * 0.5
        except:
            stress_factors += 1
    
    if "Social_Support" in answers:
        try:
            social_val = float(answers["Social_Support"]) if isinstance(answers["Social_Support"], (int, float)) else 3
            stress_factors += (5 - social_val) * 0.5
        except:
            stress_factors += 1
    
    if "Counseling" in answers:
        counseling_val = str(answers.get("Counseling", "No")).lower()
        if "no" in counseling_val:
            stress_factors += 1.5
    
    stress_level = min(5, max(0, int(stress_factors / 2)))
    return stress_level


def get_question_for_field(field_name: str) -> str:
    """Get conversational question for a given field."""
    questions = {
        "Age": "🎂 What's your age group?",
        "Course": "📚 What's your course/major?",
        "Gender": "👤 How do you identify?",
        "CGPA": "📊 What's your current GPA/CGPA?",
        "Sleep_Quality": "😴 How's your sleep quality?",
        "Physical_Activity": "🏃 How active are you physically?",
        "Diet_Quality": "🥗 How's your diet quality?",
        "Social_Support": "👥 Do you have good social support?",
        "Relationship": "❤️ How's your relationship status?",
        "Substance_Use": "🚭 Do you use substances (caffeine, etc)?",
        "Counseling": "🗣️ Are you in counseling?",
        "Family_History": "👨‍👩‍👧 Any family history of mental health issues?",
        "Chronic_Illness": "🏥 Do you have any chronic illness?",
        "Financial_Stress": "💰 How's your financial situation?",
        "Extracurricular": "🎨 Are you involved in extracurricular activities?",
        "Semester": "📅 What semester are you in?",
        "Residence_Type": "🏠 Where do you stay?",
    }
    return questions.get(field_name, f"Tell me about {field_name}")


def get_field_options(field_name: str) -> list:
    """Get predefined button options for each field."""
    options = {
        "Age": ["18-20", "21-23", "24-26", "27-30", "30+"],
        "Gender": ["🧑 Male", "👩 Female", "🧑‍🤝‍🧑 Other"],
        "Course": ["CS", "Business", "Engineering", "Medicine", "Arts", "Science"],
        "CGPA": ["3.5-4.0", "3.0-3.5", "2.5-3.0", "2.0-2.5", "Below 2.0"],
        "Sleep_Quality": ["😴 Poor (1)", "😴 Fair (2)", "😴 Good (3)", "😴 Very Good (4)", "😴 Excellent (5)"],
        "Physical_Activity": ["🔴 None (1)", "🟠 Minimal (2)", "🟡 Moderate (3)", "🟢 Active (4)", "💚 Very Active (5)"],
        "Diet_Quality": ["🔴 Poor (1)", "🟠 Fair (2)", "🟡 Good (3)", "🟢 Very Good (4)", "💚 Excellent (5)"],
        "Social_Support": ["❌ None (1)", "⚠️ Weak (2)", "😐 Moderate (3)", "✅ Good (4)", "💜 Excellent (5)"],
        "Relationship": ["Single", "In a relationship", "Married", "Complicated", "Prefer not to say"],
        "Substance_Use": ["None", "Occasionally", "Regularly", "Daily"],
        "Counseling": ["Yes, actively", "Previously", "Open to it", "No"],
        "Family_History": ["Yes - Depression", "Yes - Anxiety", "Yes - Other", "No", "Not sure"],
        "Chronic_Illness": ["Yes", "No", "Under investigation"],
        "Financial_Stress": ["🔴 Very High (1)", "🟠 High (2)", "🟡 Moderate (3)", "🟢 Low (4)", "💚 None (5)"],
        "Extracurricular": ["Very involved", "Somewhat involved", "Minimally involved", "Not involved"],
        "Semester": ["1st", "2nd", "3rd", "4th", "5th+"],
        "Residence_Type": ["Home", "Hostel", "Apartment", "Dorm", "Other"],
    }
    return options.get(field_name, [])


# =============================================
# Persistence layer (SQLite + optional Supabase)
# =============================================
def get_db_connection() -> sqlite3.Connection:
    """Get SQLite connection for AuraCheck app data."""
    os.makedirs(DB_DIR, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON;")
    return connection


def is_supabase_enabled() -> bool:
    """Return True when Supabase env variables are configured."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


def is_dns_resolution_error(exception: Exception) -> bool:
    """Return True when exception text indicates DNS hostname resolution failed."""
    error_text = str(exception).lower()
    patterns = [
        "name or service not known",
        "temporary failure in name resolution",
        "nodename nor servname provided",
        "failed to resolve",
        "getaddrinfo failed",
    ]
    return any(pattern in error_text for pattern in patterns)


@st.cache_resource
def get_supabase_client():
    """Create and cache Supabase client."""
    if st.session_state.get("supabase_sync_temporarily_disabled"):
        return None
    if not is_supabase_enabled():
        return None
    try:
        parsed = urlparse(SUPABASE_URL)
        if not parsed.scheme or not parsed.netloc:
            return None
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


def sync_payload_to_supabase(table_name: str, payload: dict, on_conflict: str) -> None:
    """Sync payload to Supabase with graceful fallback to local-only mode."""
    client = get_supabase_client()
    if client is None:
        return
    try:
        client.table(table_name).upsert(payload, on_conflict=on_conflict).execute()
        st.session_state["last_supabase_sync_error"] = None
    except Exception as exc:
        if is_dns_resolution_error(exc):
            st.session_state["supabase_sync_temporarily_disabled"] = True
            st.session_state["last_supabase_sync_error"] = (
                "Supabase host could not be resolved. Local save succeeded; "
                "remote sync paused for this session."
            )
            return
        st.session_state["last_supabase_sync_error"] = f"{table_name} sync failed: {exc}"


def sync_user_to_supabase(user_payload: dict) -> None:
    """Mirror user record to Supabase when configured."""
    sync_payload_to_supabase("users", user_payload, "user_id")


def sync_profile_to_supabase(profile_payload: dict) -> None:
    """Mirror profile record to Supabase when configured."""
    sync_payload_to_supabase("profile", profile_payload, "user_id")


def sync_daily_input_to_supabase(daily_payload: dict) -> None:
    """Mirror daily input record to Supabase when configured."""
    sync_payload_to_supabase("daily_inputs", daily_payload, "user_id,input_date")


def init_database() -> None:
    """Create required tables if they do not exist."""
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                phone_number TEXT,
                city TEXT,
                zip_code TEXT,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                is_verified INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS profile (
                profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                age INTEGER,
                lifestyle_parameters TEXT,
                personal_details TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_inputs (
                entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                input_date TEXT NOT NULL,
                submitted_at TEXT NOT NULL,
                answers_json TEXT NOT NULL,
                prediction_json TEXT NOT NULL,
                cluster INTEGER,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE(user_id, input_date)
            );
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS daily_feedback (
                feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                input_date TEXT NOT NULL,
                recommendation_followed INTEGER,
                recommendation_helpful INTEGER,
                feedback_rating INTEGER,
                app_feedback TEXT,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE(user_id, input_date)
            );
            """
        )

        # Backward-compatible migration for older local DBs.
        cursor.execute("PRAGMA table_info(users)")
        existing_user_columns = {row[1] for row in cursor.fetchall()}
        for column_name in STATIC_USER_FIELDS.values():
            if column_name not in existing_user_columns:
                cursor.execute(f"ALTER TABLE users ADD COLUMN {column_name} TEXT")

        connection.commit()


    # =====================================
    # Authentication and profile data access
    # =====================================
def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """Hash password with PBKDF2-HMAC-SHA256 and a per-user salt."""
    salt_value = salt or secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_value.encode("utf-8"),
        200000,
    )
    return hashed.hex(), salt_value


def verify_password(password: str, expected_hash: str, salt: str) -> bool:
    """Verify password against stored hash and salt."""
    calculated_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(calculated_hash, expected_hash)


def create_user(
    first_name: str,
    last_name: str,
    email: str,
    password: str,
    phone_number: str = "",
    city: str = "",
    zip_code: str = "",
) -> tuple[bool, str]:
    """Create a user account in SQL database."""
    normalized_email = email.strip().lower()
    password_hash, password_salt = hash_password(password)
    user_id = str(uuid.uuid4())

    try:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO users (
                    user_id, first_name, last_name, email,
                    phone_number, city, zip_code,
                    password_hash, password_salt, is_verified
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    user_id,
                    first_name.strip(),
                    last_name.strip(),
                    normalized_email,
                    phone_number.strip() or None,
                    city.strip() or None,
                    zip_code.strip() or None,
                    password_hash,
                    password_salt,
                ),
            )
            connection.commit()

        sync_user_to_supabase(
            {
                "user_id": user_id,
                "first_name": first_name.strip(),
                "last_name": last_name.strip(),
                "email": normalized_email,
                "phone_number": phone_number.strip() or None,
                "city": city.strip() or None,
                "zip_code": zip_code.strip() or None,
                "password_hash": password_hash,
                "password_salt": password_salt,
                "is_verified": False,
            }
        )
        return True, user_id
    except sqlite3.IntegrityError:
        return False, "A user with this email already exists."
    except Exception:
        return False, "Unable to create account right now. Please try again."


def authenticate_user(email: str, password: str) -> tuple[bool, Optional[dict], str]:
    """Authenticate user by email and password."""
    normalized_email = email.strip().lower()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT user_id, first_name, last_name, email, password_hash, password_salt
            FROM users
            WHERE email = ?
            """,
            (normalized_email,),
        )
        row = cursor.fetchone()

    if not row:
        return False, None, "No account found with this email."

    user_data = {
        "user_id": row[0],
        "first_name": row[1],
        "last_name": row[2],
        "email": row[3],
    }
    if not verify_password(password, row[4], row[5]):
        return False, None, "Invalid password."

    return True, user_data, ""


def upsert_profile(user_id: str, age: Optional[int], lifestyle_parameters: str, personal_details: str) -> bool:
    """Create or update profile details for a user."""
    try:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO profile (user_id, age, lifestyle_parameters, personal_details, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id) DO UPDATE SET
                    age = excluded.age,
                    lifestyle_parameters = excluded.lifestyle_parameters,
                    personal_details = excluded.personal_details,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    age,
                    lifestyle_parameters.strip() or None,
                    personal_details.strip() or None,
                ),
            )
            connection.commit()

        sync_profile_to_supabase(
            {
                "user_id": user_id,
                "age": age,
                "lifestyle_parameters": {"text": lifestyle_parameters.strip()} if lifestyle_parameters.strip() else {},
                "personal_details": {"text": personal_details.strip()} if personal_details.strip() else {},
            }
        )
        return True
    except Exception:
        return False


def get_user_static_answers(user_id: str) -> dict:
    """Return saved baseline answers (age/course/gender) for a user, if present."""
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT survey_age, survey_course, survey_gender
            FROM users
            WHERE user_id = ?
            LIMIT 1
            """,
            (user_id,),
        )
        row = cursor.fetchone()

    if not row:
        return {}

    return {
        "Age": row[0],
        "Course": row[1],
        "Gender": row[2],
    }


def save_user_static_answer_if_missing(user_id: str, field_name: str, field_value: str) -> None:
    """Persist first submitted baseline answer only once for the user."""
    column_name = STATIC_USER_FIELDS.get(field_name)
    value = (field_value or "").strip()
    if not column_name or not value:
        return

    with get_db_connection() as connection:
        connection.execute(
            f"""
            UPDATE users
            SET {column_name} = COALESCE(NULLIF({column_name}, ''), ?),
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ?
            """,
            (value, user_id),
        )
        connection.commit()


def save_user_daily_input_to_sql(user_id: str, answers: dict, prediction: dict, cluster: int) -> tuple[bool, str]:
    """Save one daily questionnaire response per user to SQL."""
    today_value = date.today().isoformat()
    submitted_at = datetime.now().isoformat()

    try:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO daily_inputs (
                    user_id, input_date, submitted_at, answers_json, prediction_json, cluster
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    today_value,
                    submitted_at,
                    json.dumps(answers),
                    json.dumps(prediction),
                    cluster,
                ),
            )
            connection.commit()

        sync_daily_input_to_supabase(
            {
                "user_id": user_id,
                "input_date": today_value,
                "submitted_at": submitted_at,
                "answers_json": answers,
                "prediction_json": prediction,
                "cluster": cluster,
            }
        )
        st.session_state["last_local_sql_error"] = None
        return True, ""
    except sqlite3.IntegrityError:
        return False, "You have already submitted today's input. Please come back tomorrow."
    except Exception as exc:
        st.session_state["last_local_sql_error"] = f"daily_inputs local save failed: {exc}"
        return False, "Unable to save your daily input right now."


def has_user_submitted_today(user_id: str) -> bool:
    """Check whether user already submitted daily survey today."""
    today_value = date.today().isoformat()
    with get_db_connection() as connection:
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT 1
            FROM daily_inputs
            WHERE user_id = ? AND input_date = ?
            LIMIT 1
            """,
            (user_id, today_value),
        )
        return cursor.fetchone() is not None


def upsert_daily_feedback(
    user_id: str,
    input_date: str,
    recommendation_followed: Optional[bool],
    recommendation_helpful: Optional[bool],
    feedback_rating: Optional[int],
    app_feedback: str,
) -> tuple[bool, str]:
    """Create or update per-day recommendation/app feedback for a user."""
    try:
        with get_db_connection() as connection:
            connection.execute(
                """
                INSERT INTO daily_feedback (
                    user_id, input_date, recommendation_followed,
                    recommendation_helpful, feedback_rating, app_feedback, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(user_id, input_date) DO UPDATE SET
                    recommendation_followed = excluded.recommendation_followed,
                    recommendation_helpful = excluded.recommendation_helpful,
                    feedback_rating = excluded.feedback_rating,
                    app_feedback = excluded.app_feedback,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    user_id,
                    input_date,
                    None if recommendation_followed is None else int(recommendation_followed),
                    None if recommendation_helpful is None else int(recommendation_helpful),
                    feedback_rating,
                    app_feedback.strip() or None,
                ),
            )
            connection.commit()
        return True, ""
    except Exception:
        return False, "Unable to save feedback right now."


def get_user_daily_history(user_id: str) -> pd.DataFrame:
    """Fetch a user's day-by-day survey history with optional feedback."""
    with get_db_connection() as connection:
        history_df = pd.read_sql_query(
            """
            SELECT
                d.entry_id,
                d.user_id,
                d.input_date,
                d.submitted_at,
                d.prediction_json,
                d.cluster,
                f.recommendation_followed,
                f.recommendation_helpful,
                f.feedback_rating,
                f.app_feedback
            FROM daily_inputs d
            LEFT JOIN daily_feedback f
              ON d.user_id = f.user_id
             AND d.input_date = f.input_date
            WHERE d.user_id = ?
            ORDER BY d.input_date ASC
            """,
            connection,
            params=(user_id,),
        )

    if history_df.empty:
        return history_df

    parsed_predictions = history_df["prediction_json"].apply(parse_prediction_json)
    history_df["stress_level"] = parsed_predictions.apply(lambda p: p.get("stress_level"))
    history_df["anxiety_score"] = parsed_predictions.apply(lambda p: p.get("anxiety_score"))
    history_df["depression_score"] = parsed_predictions.apply(lambda p: p.get("depression_score"))
    history_df["mental_health_pct"] = parsed_predictions.apply(lambda p: p.get("mental_health_pct"))
    return history_df


def parse_prediction_json(prediction_json: str) -> dict:
    """Safely parse prediction JSON payload."""
    try:
        if not prediction_json:
            return {}
        parsed = json.loads(prediction_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def get_admin_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch users and daily input records for admin view."""
    with get_db_connection() as connection:
        users_df = pd.read_sql_query(
            """
            SELECT
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone_number,
                u.city,
                u.zip_code,
                u.created_at,
                COUNT(d.entry_id) AS total_entries,
                MAX(d.input_date) AS last_input_date
            FROM users u
            LEFT JOIN daily_inputs d ON u.user_id = d.user_id
            GROUP BY
                u.user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.phone_number,
                u.city,
                u.zip_code,
                u.created_at
            ORDER BY u.created_at DESC
            """,
            connection,
        )

        daily_df = pd.read_sql_query(
            """
            SELECT
                entry_id,
                user_id,
                input_date,
                submitted_at,
                prediction_json,
                cluster
            FROM daily_inputs
            ORDER BY submitted_at DESC
            """,
            connection,
        )

    if not daily_df.empty:
        parsed_predictions = daily_df["prediction_json"].apply(parse_prediction_json)
        daily_df["stress_level"] = parsed_predictions.apply(lambda p: p.get("stress_level"))
        daily_df["anxiety_score"] = parsed_predictions.apply(lambda p: p.get("anxiety_score"))
        daily_df["depression_score"] = parsed_predictions.apply(lambda p: p.get("depression_score"))

    return users_df, daily_df


# ==============================
# UI renderers and app navigation
# ==============================
def render_admin_page() -> None:
    """Render a simple admin page for users and daily trends."""
    st.markdown("<div class='content-placeholder'>", unsafe_allow_html=True)
    st.markdown("<div class='middle-section'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'>Admin View</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>Users and Daily History Trends</h2>", unsafe_allow_html=True)

    users_df, daily_df = get_admin_data()

    total_users = int(len(users_df))
    total_entries = int(len(daily_df))
    today_entries = int((daily_df["input_date"] == date.today().isoformat()).sum()) if not daily_df.empty else 0

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    with metric_col_1:
        st.metric("Total Users", total_users)
    with metric_col_2:
        st.metric("Total Daily Entries", total_entries)
    with metric_col_3:
        st.metric("Today's Entries", today_entries)

    st.markdown("#### 👥 Users")
    if users_df.empty:
        st.info("No users found yet.")
    else:
        users_display = users_df[[
            "first_name", "last_name", "email", "phone_number", "city", "zip_code", "total_entries", "last_input_date"
        ]].copy()
        users_display = users_display.rename(
            columns={
                "first_name": "First Name",
                "last_name": "Last Name",
                "email": "Email",
                "phone_number": "Phone",
                "city": "City",
                "zip_code": "ZIP",
                "total_entries": "Entries",
                "last_input_date": "Last Input Date",
            }
        )
        st.dataframe(users_display, hide_index=True)

    st.markdown("#### 📈 Daily Trend")
    if daily_df.empty:
        st.info("No daily input history found yet.")
    else:
        trend_df = (
            daily_df.groupby("input_date", as_index=False)
            .agg(submissions=("entry_id", "count"), avg_stress=("stress_level", "mean"))
            .sort_values("input_date")
        )

        trend_fig = go.Figure()
        trend_fig.add_trace(
            go.Bar(
                x=trend_df["input_date"],
                y=trend_df["submissions"],
                name="Submissions",
                marker_color="#5B7FEA",
                yaxis="y",
            )
        )
        trend_fig.add_trace(
            go.Scatter(
                x=trend_df["input_date"],
                y=trend_df["avg_stress"],
                mode="lines+markers",
                name="Avg Stress",
                marker_color="#9B7FB5",
                yaxis="y2",
            )
        )
        trend_fig.update_layout(
            xaxis_title="Date",
            yaxis=dict(title="Submissions"),
            yaxis2=dict(title="Avg Stress", overlaying="y", side="right", range=[0, 5]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=360,
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(trend_fig, width="stretch")

        st.markdown("#### 🧾 User Daily History")
        user_option_map = {
            f"{row['first_name']} {row['last_name']} ({row['email']})": row["user_id"]
            for _, row in users_df.iterrows()
        }
        selected_user_label = st.selectbox("Select user", options=list(user_option_map.keys()))
        selected_user_id = user_option_map[selected_user_label]

        user_history_df = daily_df[daily_df["user_id"] == selected_user_id].copy()
        user_history_df = user_history_df.sort_values("submitted_at")

        if user_history_df.empty:
            st.info("This user has no daily entries yet.")
        else:
            user_stress_fig = go.Figure()
            user_stress_fig.add_trace(
                go.Scatter(
                    x=user_history_df["input_date"],
                    y=user_history_df["stress_level"],
                    mode="lines+markers",
                    name="Stress Level",
                    marker_color="#3F2456",
                )
            )
            user_stress_fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Stress Level",
                yaxis=dict(range=[0, 5]),
                height=300,
                margin=dict(l=20, r=20, t=20, b=20),
            )
            st.plotly_chart(user_stress_fig, width="stretch")

            history_display = user_history_df[[
                "input_date", "submitted_at", "stress_level", "anxiety_score", "depression_score", "cluster"
            ]].copy()
            history_display = history_display.rename(
                columns={
                    "input_date": "Date",
                    "submitted_at": "Submitted At",
                    "stress_level": "Stress",
                    "anxiety_score": "Anxiety %",
                    "depression_score": "Depression %",
                    "cluster": "Cluster",
                }
            )
            st.dataframe(history_display, hide_index=True)

    if st.button("← Back to AuraCheck", key="admin_back_to_main", width="stretch"):
        st.session_state["auth_page"] = "main"
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def save_user_response_to_json(answers: dict, prediction: dict, cluster: int) -> None:
    """Save user response to JSON file."""
    try:
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "user_inputs": answers,
            "predictions": prediction,
            "cluster": cluster
        }
        
        all_responses = []
        
        if os.path.exists(USER_RESPONSES_JSON_PATH):
            with open(USER_RESPONSES_JSON_PATH, 'r') as f:
                try:
                    all_responses = json.load(f)
                except:
                    all_responses = []
        
        all_responses.append(response_data)
        
        with open(USER_RESPONSES_JSON_PATH, 'w') as f:
            json.dump(all_responses, f, indent=2)
        
    except Exception:
        # JSON export is optional and should not block the UI flow.
        pass


def initialize_state() -> None:
    """Initialize session state."""
    defaults = {
        "last_answers": {},
        "last_prediction": None,
        "last_cluster": None,
        "show_results": False,
        "auth_page": "main",
        "current_user_id": None,
        "current_user_email": None,
        "current_user_name": None,
        "last_local_sql_error": None,
        "last_supabase_sync_error": None,
        "supabase_sync_temporarily_disabled": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_survey_state() -> None:
    """Reset questionnaire and latest result state."""
    st.session_state["last_answers"] = {}
    st.session_state["last_prediction"] = None
    st.session_state["last_cluster"] = None
    st.session_state["show_results"] = False


def render_user_progress_section(user_id: str) -> None:
    """Render per-user day-by-day progress charts, insights, and feedback form."""
    history_df = get_user_daily_history(user_id)

    st.markdown("#### 📈 Your Day-by-Day Progress")
    if history_df.empty:
        st.info("No saved daily history yet. Complete today's survey to start tracking progress.")
        return

    trend_fig = go.Figure()
    trend_fig.add_trace(
        go.Scatter(
            x=history_df["input_date"],
            y=history_df["stress_level"],
            mode="lines+markers",
            name="Stress",
            marker_color="#9B7FB5",
            yaxis="y",
        )
    )
    trend_fig.add_trace(
        go.Scatter(
            x=history_df["input_date"],
            y=history_df["mental_health_pct"],
            mode="lines+markers",
            name="Mental Health %",
            marker_color="#5B7FEA",
            yaxis="y2",
        )
    )
    trend_fig.update_layout(
        xaxis_title="Date",
        yaxis=dict(title="Stress", range=[0, 5]),
        yaxis2=dict(title="Mental Health %", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(trend_fig, width="stretch")

    latest = history_df.iloc[-1]
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Latest Stress", f"{float(latest['stress_level']):.1f}/5" if pd.notna(latest["stress_level"]) else "N/A")
    with col_b:
        st.metric("Latest Mental Health", f"{float(latest['mental_health_pct']):.1f}%" if pd.notna(latest["mental_health_pct"]) else "N/A")
    with col_c:
        st.metric("Entries", str(len(history_df)))

    if len(history_df) >= 2:
        prev = history_df.iloc[-2]
        stress_improvement = (float(prev["stress_level"]) - float(latest["stress_level"])) if pd.notna(prev["stress_level"]) and pd.notna(latest["stress_level"]) else 0.0
        wellbeing_change = (float(latest["mental_health_pct"]) - float(prev["mental_health_pct"])) if pd.notna(prev["mental_health_pct"]) and pd.notna(latest["mental_health_pct"]) else 0.0

        if stress_improvement > 0:
            st.success(f"✅ Improvement: stress improved by {stress_improvement:.1f} vs previous day.")
        elif stress_improvement < 0:
            st.warning(f"⚠️ Stress increased by {abs(stress_improvement):.1f} vs previous day.")
        else:
            st.info("ℹ️ Stress is unchanged vs previous day.")

        if wellbeing_change > 0:
            st.success(f"✅ Mental health score improved by {wellbeing_change:.1f}% vs previous day.")
        elif wellbeing_change < 0:
            st.warning(f"⚠️ Mental health score decreased by {abs(wellbeing_change):.1f}% vs previous day.")

    effectiveness_df = history_df.dropna(subset=["recommendation_followed", "stress_level"])
    if len(effectiveness_df) >= 3:
        followed = effectiveness_df[effectiveness_df["recommendation_followed"] == 1]
        not_followed = effectiveness_df[effectiveness_df["recommendation_followed"] == 0]
        if not followed.empty and not not_followed.empty:
            followed_avg = float(followed["stress_level"].mean())
            not_followed_avg = float(not_followed["stress_level"].mean())
            if followed_avg < not_followed_avg:
                st.success("✅ Days marked as following recommendations show lower average stress.")
            elif followed_avg > not_followed_avg:
                st.info("ℹ️ Recommendation effect is not yet clear. Keep tracking daily for stronger signal.")

    st.markdown("#### 📝 Recommendation & App Feedback")
    today_value = date.today().isoformat()
    today_row = history_df[history_df["input_date"] == today_value]
    if today_row.empty:
        st.info("Submit today's survey first, then share whether recommendations helped.")
    else:
        existing = today_row.iloc[-1]

        follow_default = "Not yet"
        if pd.notna(existing.get("recommendation_followed")):
            follow_default = "Yes" if int(existing["recommendation_followed"]) == 1 else "No"

        helpful_default = "Not sure"
        if pd.notna(existing.get("recommendation_helpful")):
            helpful_default = "Yes" if int(existing["recommendation_helpful"]) == 1 else "No"

        rating_default = int(existing["feedback_rating"]) if pd.notna(existing.get("feedback_rating")) else 3
        comment_default = str(existing["app_feedback"]) if pd.notna(existing.get("app_feedback")) else ""

        with st.form("daily_feedback_form"):
            followed_choice = st.radio(
                "Did you follow today's recommendations?",
                options=["Yes", "No", "Not yet"],
                index=["Yes", "No", "Not yet"].index(follow_default),
                horizontal=True,
            )
            helpful_choice = st.radio(
                "Are the recommendations helping?",
                options=["Yes", "No", "Not sure"],
                index=["Yes", "No", "Not sure"].index(helpful_default),
                horizontal=True,
            )
            feedback_rating = st.slider("App experience rating (1-5)", min_value=1, max_value=5, value=rating_default)
            app_feedback = st.text_area("Feedback for app improvements", value=comment_default)
            feedback_submit = st.form_submit_button("Save Feedback", width="stretch")

        if feedback_submit:
            followed_value = True if followed_choice == "Yes" else False if followed_choice == "No" else None
            helpful_value = True if helpful_choice == "Yes" else False if helpful_choice == "No" else None
            saved, message = upsert_daily_feedback(
                user_id=user_id,
                input_date=today_value,
                recommendation_followed=followed_value,
                recommendation_helpful=helpful_value,
                feedback_rating=feedback_rating,
                app_feedback=app_feedback,
            )
            if saved:
                st.success("✅ Feedback saved. Thanks for helping improve AuraCheck.")
            else:
                st.warning(f"⚠️ {message}")


def is_admin_user() -> bool:
    """Return True only for the allowed admin email."""
    current_email = (st.session_state.get("current_user_email") or "").strip().lower()
    return current_email == ADMIN_EMAIL


def all_required_answered(answers: dict, required_fields: list) -> bool:
    """Check that all required fields are answered with valid non-skipped values."""
    for field in required_fields:
        value = answers.get(field)
        if value is None:
            return False
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized or normalized.lower() == "skipped":
                return False
    return True


def render_auth_page(auth_page: str) -> None:
    """Render dedicated authentication pages."""
    st.markdown("<div class='content-placeholder'>", unsafe_allow_html=True)
    st.markdown("<div class='middle-section'>", unsafe_allow_html=True)
    st.image("Dataset/logo.jpg", width=90)

    if auth_page == "signup":
        st.markdown("<h1 style='text-align: center;'>Join AuraCheck</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Create your account</h2>", unsafe_allow_html=True)

        with st.form("signup_form"):
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            signup_email = st.text_input("Email Address")
            phone_number = st.text_input("Phone Number (Optional)")
            city = st.text_input("City (Optional)")
            zip_code = st.text_input("ZIP (Optional)")
            signup_password = st.text_input("Password", type="password")
            signup_confirm_password = st.text_input("Confirm Password", type="password")
            signup_submit = st.form_submit_button("Sign Up", width="stretch")

        if signup_submit:
            if not first_name.strip() or not last_name.strip() or not signup_email.strip():
                st.warning("⚠️ Please fill in first name, last name, and email address.")
            elif "@" not in signup_email or "." not in signup_email:
                st.warning("⚠️ Please enter a valid email address.")
            elif len(signup_password) < 8:
                st.warning("⚠️ Password must be at least 8 characters.")
            elif signup_password != signup_confirm_password:
                st.warning("⚠️ Password and confirm password do not match.")
            else:
                created, message = create_user(
                    first_name=first_name,
                    last_name=last_name,
                    email=signup_email,
                    password=signup_password,
                    phone_number=phone_number,
                    city=city,
                    zip_code=zip_code,
                )
                if created:
                    st.success(f"✅ Verification link has been sent to {signup_email.strip().lower()}.")
                else:
                    st.warning(f"⚠️ {message}")

        if st.button("Already have an account? Log In", key="goto_login", width="stretch"):
            st.session_state["auth_page"] = "login"
            st.rerun()

    if auth_page == "login":
        st.markdown("<h1 style='text-align: center;'>Login</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Access your AuraCheck account</h2>", unsafe_allow_html=True)

        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submit = st.form_submit_button("Log In", width="stretch")

        if login_submit:
            if not login_email.strip() or not login_password.strip():
                st.warning("⚠️ Please enter your email and password.")
            else:
                is_valid, user_data, auth_message = authenticate_user(login_email, login_password)
                if is_valid and user_data:
                    st.session_state["current_user_id"] = user_data["user_id"]
                    st.session_state["current_user_email"] = user_data["email"]
                    st.session_state["current_user_name"] = f"{user_data['first_name']} {user_data['last_name']}"
                    st.session_state["auth_page"] = "profile"
                    st.success("✅ Login successful.")
                    st.rerun()
                else:
                    st.warning(f"⚠️ {auth_message}")

        st.markdown("<h4 style='text-align: center;'>Forgot Password?</h4>", unsafe_allow_html=True)
        forgot_email = st.text_input("Email for password reset", key="forgot_email")
        if st.button("Send Reset Link", key="forgot_password_btn", width="stretch"):
            if not forgot_email.strip():
                st.warning("⚠️ Please enter your email address.")
            elif "@" not in forgot_email or "." not in forgot_email:
                st.warning("⚠️ Please enter a valid email address.")
            else:
                st.success(f"✅ Reset link has been sent to {forgot_email.strip()}.")

        if st.button("Need an account? Sign Up", key="goto_signup", width="stretch"):
            st.session_state["auth_page"] = "signup"
            st.rerun()

    if st.button("← Back to AuraCheck", key="back_to_main", width="stretch"):
        st.session_state["auth_page"] = "main"
        st.rerun()

    if auth_page == "profile":
        st.markdown("<h1 style='text-align: center;'>Profile</h1>", unsafe_allow_html=True)
        current_name = st.session_state.get("current_user_name") or "User"
        current_email = st.session_state.get("current_user_email") or ""
        current_user_id = st.session_state.get("current_user_id")
        st.markdown(f"<h2 style='text-align: center;'>Welcome, {current_name}</h2>", unsafe_allow_html=True)
        if current_email:
            st.markdown(f"<p style='text-align: center;'>Logged in as {current_email}</p>", unsafe_allow_html=True)

        saved_static_answers = get_user_static_answers(current_user_id) if current_user_id else {}
        static_age = (saved_static_answers.get("Age") or "").strip()
        static_course = (saved_static_answers.get("Course") or "").strip()
        static_gender = (saved_static_answers.get("Gender") or "").strip()

        if static_age or static_course or static_gender:
            st.markdown("#### Baseline Assessment Details")
            st.caption("These values are reused automatically in future assessments.")
            static_col_1, static_col_2, static_col_3 = st.columns(3)
            with static_col_1:
                st.text_input("Age", value=static_age or "Not set", disabled=True)
            with static_col_2:
                st.text_input("Course", value=static_course or "Not set", disabled=True)
            with static_col_3:
                st.text_input("Gender", value=static_gender or "Not set", disabled=True)

        if st.button("Continue to AuraCheck", key="profile_to_main", width="stretch"):
            st.session_state["auth_page"] = "main"
            st.rerun()

        if st.button("Log Out", key="logout_btn", width="stretch"):
            st.session_state["current_user_id"] = None
            st.session_state["current_user_email"] = None
            st.session_state["current_user_name"] = None
            st.session_state["auth_page"] = "main"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    """Main Streamlit Application."""

    # Initialize required runtime state before rendering any page sections.
    initialize_state()
    init_database()

    if st.session_state.get("auth_page") == "admin":
        if is_admin_user():
            render_admin_page()
            return
        st.session_state["auth_page"] = "main"
        st.warning("⚠️ Admin view is restricted to authorized admin only.")

    if st.session_state.get("auth_page") in {"signup", "login", "profile"}:
        render_auth_page(st.session_state.get("auth_page"))
        return
    
    # --- CONTENT PLACEHOLDER - Shiny purple background ---
    st.markdown("<div class='content-placeholder'>", unsafe_allow_html=True)
    
    # --- CREATE 3-COLUMN LAYOUT ---
    left_col, middle_col, right_col = st.columns([1.15, 2.25, 0.95], gap="large")
    
    # ========== LEFT COLUMN: Header & Positive Thoughts ==========
    with left_col:
        logo_pad_top, logo_mid, logo_pad_bottom = st.columns([1.2, 2, 0.8])
        with logo_mid:
            st.image("Dataset/logo.jpg", width=90)
        st.markdown("<h1 style='text-align: center;'>AuraCheck</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Your Quick Wellbeing Check-in</h2>", unsafe_allow_html=True)
        
        # Good Thoughts Section
        st.markdown("<div class='good-thoughts-header'>✨ Thought for the Day: </div>", unsafe_allow_html=True)
        
        thoughts_js = json.dumps(POSITIVE_THOUGHTS)
        components.html(
            f"""
            <div style="display:flex; justify-content:center; width:100%; margin-top: 6px;">
                <div id="thought-card" style="
                    width: 92%;
                    min-height: 140px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    text-align:center;
                    color:#4A2F66;
                    font-size:30px;
                    font-weight:700;
                    line-height:1.45;
                    padding:18px 20px;
                    border-radius:14px;
                    border:2px solid rgba(155,127,181,0.35);
                    background: rgba(255,255,255,0.32);
                    box-sizing: border-box;
                    transition: opacity 500ms ease-in-out;
                "></div>
            </div>
            <script>
                const thoughts = {thoughts_js};
                const thoughtCard = document.getElementById('thought-card');
                let lastIndex = -1;

                function nextThought() {{
                    if (!thoughtCard || thoughts.length === 0) return;
                    thoughtCard.style.opacity = 0;
                    setTimeout(() => {{
                        let index = Math.floor(Math.random() * thoughts.length);
                        if (thoughts.length > 1) {{
                            while (index === lastIndex) {{
                                index = Math.floor(Math.random() * thoughts.length);
                            }}
                        }}
                        lastIndex = index;
                        thoughtCard.textContent = thoughts[index];
                        thoughtCard.style.transition = 'opacity 500ms ease-in-out';
                        thoughtCard.style.opacity = 1;
                    }}, 450);
                }}

                nextThought();
                setInterval(nextThought, 3200);
            </script>
            """,
            height=190,
        )
    
    # ========== MIDDLE COLUMN: Questions & Analysis ==========
    with middle_col:
        st.markdown("<div class='middle-section'>", unsafe_allow_html=True)
        st.markdown("<div class='middle-panel'>", unsafe_allow_html=True)
        
        required_fields = REQUIRED_FIELDS
        
        current_question_idx = len(st.session_state.get("last_answers", {}))
        answers = dict(st.session_state.get("last_answers", {}))
        current_user_id = st.session_state.get("current_user_id")

        # For logged-in users, preload baseline answers and skip re-asking them.
        if current_user_id:
            saved_static_answers = get_user_static_answers(current_user_id)
            for field_name in STATIC_USER_FIELDS:
                saved_value = (saved_static_answers.get(field_name) or "").strip()
                if saved_value and not answers.get(field_name):
                    answers[field_name] = saved_value
            if answers != st.session_state.get("last_answers", {}):
                st.session_state["last_answers"] = answers

        current_question_idx = len(answers)
        already_submitted_today = bool(current_user_id and has_user_submitted_today(current_user_id))

        if already_submitted_today:
            st.info("✅ You already submitted today's survey. Come back tomorrow for your next check-in.")
        
        # Progress Section
        if current_question_idx > 0:
            progress_pct = current_question_idx / len(required_fields)
            st.progress(progress_pct)
            st.markdown(f"<p class='progress-text'>Question {current_question_idx} of {len(required_fields)}</p>", unsafe_allow_html=True)
        
        # Questions Display
        st.markdown("<div class='questions-section'>", unsafe_allow_html=True)
        
        if current_question_idx < len(required_fields) and not already_submitted_today:
            current_field = required_fields[current_question_idx]
            options = get_field_options(current_field)
            question = get_question_for_field(current_field)
            
            st.markdown(f"<div class='question-text'>{question}</div>", unsafe_allow_html=True)
            
            if options:
                for idx, option in enumerate(options):
                    if st.button(option, key=f"btn_{current_field}_{idx}", width="stretch"):
                        clean_value = option.split("(")[0].strip() if "(" in option else option
                        clean_value = ''.join(c for c in clean_value if ord(c) < 0x1F600 or ord(c) > 0x1F64F)
                        clean_value = clean_value.strip()
                        answers[current_field] = clean_value if clean_value else option

                        # Save first baseline response permanently for logged-in users.
                        if current_user_id and current_field in STATIC_USER_FIELDS:
                            save_user_static_answer_if_missing(current_user_id, current_field, answers[current_field])

                        st.session_state["last_answers"] = answers
                        st.rerun()
            else:
                user_input = st.text_input(f"Your answer:", key=f"input_{current_field}")
                if user_input:
                    answers[current_field] = user_input
                    st.session_state["last_answers"] = answers
                    st.rerun()
        elif not already_submitted_today:
            st.markdown("<div style='text-align: center; padding: 30px 0;'><h3 style='color: #3F2456;'>✨ All set!</h3><p style='color: #7A6B8F;'>Click below to analyze your wellbeing</p></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center; padding: 30px 0;'><h3 style='color: #3F2456;'>📅 Daily survey completed</h3><p style='color: #7A6B8F;'>Your next survey unlocks tomorrow.</p></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analyze Button
        st.markdown("<div class='analyze-section'>", unsafe_allow_html=True)
        if st.button("🔍 Analyze My Results", key="analyze_btn", width="stretch", disabled=already_submitted_today):
            if not all_required_answered(answers, required_fields):
                st.warning("⚠️ Please answer all questions first!")
            else:
                stress_level = predict_stress_level(answers)
                wellbeing_pct = 70.0 if stress_level <= 2 else 50.0 if stress_level <= 4 else 30.0
                
                if stress_level <= 1:
                    cluster = 0
                elif stress_level <= 3:
                    cluster = 1
                else:
                    cluster = 2
                
                anxiety_score = min(100, max(0, wellbeing_pct - (stress_level * 8)))
                depression_score = max(0, 100 - wellbeing_pct)
                
                prediction = {
                    "stress_level": stress_level,
                    "mental_health_pct": wellbeing_pct,
                    "anxiety_score": anxiety_score,
                    "depression_score": depression_score,
                }
                
                st.session_state["last_prediction"] = prediction
                st.session_state["last_cluster"] = cluster
                st.session_state["show_results"] = True
                
                save_user_response_to_json(answers, prediction, cluster)
                if st.session_state.get("current_user_id"):
                    saved_to_sql, save_message = save_user_daily_input_to_sql(
                        user_id=st.session_state.get("current_user_id"),
                        answers=answers,
                        prediction=prediction,
                        cluster=cluster,
                    )
                    if saved_to_sql:
                        st.success("✅ Daily input saved to local database.")
                    else:
                        st.warning(f"⚠️ {save_message}")
                else:
                    st.info("ℹ️ Log in to save daily inputs to your account history.")

        if st.button("🔄 Start New Assessment", key="reset_assessment_btn", width="stretch"):
            reset_survey_state()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Results Display
        if st.session_state.get("show_results"):
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            
            prediction = st.session_state.get("last_prediction")
            
            st.markdown("<h3 style='text-align: center;'>📊 Your Wellness Assessment</h3>", unsafe_allow_html=True)
            
            # Gauge Charts
            col_gauge1, col_gauge2 = st.columns(2)
            
            with col_gauge1:
                fig_stress = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction["stress_level"],
                    title={"text": "Stress Level (0-5)"},
                    gauge={
                        "axis": {"range": [0, 5]},
                        "bar": {"color": "#9B7FB5"},
                        "steps": [
                            {"range": [0, 1.67], "color": "#b7e4c7"},
                            {"range": [1.67, 3.33], "color": "#ffe066"},
                            {"range": [3.33, 5], "color": "#ff6f69"},
                        ],
                    },
                ))
                fig_stress.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_stress, width="stretch")
            
            with col_gauge2:
                wellbeing_value = 100 - (prediction["stress_level"] * 20)
                fig_wellbeing = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=wellbeing_value,
                    title={"text": "Wellbeing (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#5B7FEA"},
                        "steps": [
                            {"range": [0, 33], "color": "#ffcccc"},
                            {"range": [33, 66], "color": "#fff4cc"},
                            {"range": [66, 100], "color": "#ccffcc"},
                        ],
                    },
                ))
                fig_wellbeing.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig_wellbeing, width="stretch")
            
            # Metric Cards
            col_metrics = st.columns(3)
            with col_metrics[0]:
                st.metric("Stress", f"{prediction['stress_level']}/5", "0.5")
            with col_metrics[1]:
                st.metric("Anxiety", f"{prediction['anxiety_score']:.0f}%", "5%")
            with col_metrics[2]:
                st.metric("Depression", f"{prediction['depression_score']:.0f}%", "3%")
            
            # Recommendations
            st.markdown("#### 💡 Recommendations")
            
            if prediction["stress_level"] >= 4:
                st.warning("⚠️ Your stress levels are high. Please consider reaching out to a counselor.")
                st.write("Immediate actions: Take deep breaths, talk to someone you trust, practice meditation.")
            else:
                st.success("✅ Your stress levels are manageable. Keep maintaining healthy habits!")
            
            st.markdown("</div>", unsafe_allow_html=True)

        if current_user_id:
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            render_user_progress_section(current_user_id)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)
        
    
    # ========== RIGHT COLUMN: Authentication ==========
    with right_col:
        if st.session_state.get("current_user_id"):
            st.markdown("<div class='auth-header'>Logged In</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='auth-subtext'>{st.session_state.get('current_user_name')}<br/>{st.session_state.get('current_user_email')}</div>",
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button("👤 Profile", key="profile_btn", width="stretch"):
                st.session_state["auth_page"] = "profile"
                st.rerun()
            if is_admin_user():
                if st.button("🛠️ Admin View", key="admin_btn_logged_in", width="stretch"):
                    st.session_state["auth_page"] = "admin"
                    st.rerun()
            if st.button("🚪 Log Out", key="logout_main_btn", width="stretch"):
                st.session_state["current_user_id"] = None
                st.session_state["current_user_email"] = None
                st.session_state["current_user_name"] = None
                st.rerun()
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-subtext'>Daily SQL history enabled</div>", unsafe_allow_html=True)
            if st.session_state.get("last_local_sql_error"):
                st.caption(f"Local SQL note: {st.session_state.get('last_local_sql_error')}")
            if st.session_state.get("last_supabase_sync_error"):
                st.caption(f"Supabase sync note: {st.session_state.get('last_supabase_sync_error')}")
        else:
        
            st.markdown("<div class='auth-header'>Join AuraCheck</div>", unsafe_allow_html=True)
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            
            if st.button("⭐ Sign Up", key="signup_btn", width="stretch"):
                st.session_state["auth_page"] = "signup"
                st.rerun()
            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
            if st.button("📝 Log In", key="login_btn", width="stretch"):
                st.session_state["auth_page"] = "login"
                st.rerun()
            
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.markdown("<div class='auth-subtext'>Save your progress &<br/>track your journey</div>", unsafe_allow_html=True)
    
    # --- CLOSE CONTENT PLACEHOLDER ---
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
