"""AuraCheck Streamlit application.

This module intentionally keeps UI rendering, authentication, local persistence,
and optional Supabase sync together to simplify project delivery.
"""


import os
import json
import uuid
import secrets
import hashlib
import hmac
from datetime import date, datetime
from typing import Any, Optional
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
USER_RESPONSES_JSON_PATH = os.path.join(APP_DIR, "Dataset", "user_responses.json")

REQUIRED_FIELDS = ["Age", "Course", "Gender", "CGPA", "Sleep_Quality", "Physical_Activity", "Diet_Quality", "Social_Support", "Relationship", "Substance_Use", "Counseling", "Family_History", "Chronic_Illness", "Financial_Stress", "Extracurricular", "Semester", "Residence_Type"]
POSITIVE_THOUGHTS = ["🌟 You are capable of overcoming challenges", "💚 Your mental health matters and deserves attention", "🌈 Every day is a fresh opportunity for growth", "💫 You have strength within you", "🌸 Self-care is not selfish, it's essential", "⭐ Progress over perfection always", "🎯 Your feelings are valid and important", "🌊 Challenges help you grow stronger", "💡 You deserve to be happy and healthy", "🦋 Transformation starts with self-compassion"]
STATIC_USER_FIELDS = {"Age": "survey_age","Course": "survey_course","Gender": "survey_gender","Relationship": "survey_relationship","CGPA": "survey_cgpa"}

def normalize_supabase_url(url_value: str) -> str:
    """Normalize Supabase URL so env variants do not break client setup."""
    normalized = (url_value or "").strip().strip('"').strip("'").rstrip("/")
    if normalized and not normalized.startswith(("http://", "https://")):
        normalized = f"https://{normalized}"
    return normalized

SUPABASE_URL = normalize_supabase_url(os.getenv("SUPABASE_URL", ""))
SUPABASE_KEY = (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY") or "").strip()
st.set_page_config(page_title="AuraCheck", page_icon="💜", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700&display=swap');
    :root {
        --purple-1: #9B7FB5;
        --purple-2: #8B6FA5;
        --purple-3: #705291;
        --ink-1: #3F2456;
        --ink-2: #5A3D79;
        --blue-1: #355DCB;
        --blue-2: #234AAE;
    }
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    .stApp { background: linear-gradient(135deg, #E3D7F2 0%, #ECE0F7 50%, #DDCFEC 100%); min-height: 100vh; padding: 4px 10px; }
    .main .block-container { padding-top: 0.5rem; padding-bottom: 0.5rem; }
    .main {background: linear-gradient(135deg, rgba(248,244,253,.95) 0%, rgba(244,238,251,.95) 50%, rgba(239,232,248,.95) 100%);
        border-radius: 40px; padding: 0; margin: 12px auto; max-width: 750px; width: 100%; overflow: hidden;
        border: 5px solid var(--purple-3);
        box-shadow: 0 35px 100px rgba(124,91,166,.3), 0 15px 50px rgba(155,127,181,.25), inset 0 1px 0 rgba(255,255,255,.7);}
    h1, h2 { text-align: center; }
    h1 { color: var(--ink-1); font-size: 40px; margin: 8px 0 4px; font-weight: 800; font-family: 'Poppins', sans-serif; letter-spacing: -0.8px; line-height: 1.1; }
    h2 { color: var(--ink-2); font-size: 18px; font-weight: 500; margin: 0 0 14px; letter-spacing: 0.3px; }
    h3 { color: var(--ink-1); font-family: 'Poppins', sans-serif; font-size: 24px; font-weight: 700; margin: 0 0 12px; letter-spacing: -0.3px; }
    h4 { color: #3D2C55; font-size: 18px; font-weight: 600; margin-bottom: 15px; }
    p, span, label { color: #352549; font-weight: 500; line-height: 1.6; }

    .header-section { padding: 30px 28px 16px; text-align: center; }
    .questions-section, .results-section { padding: 18px 28px; background: transparent; }
    .analyze-section { padding: 14px 28px; text-align: center; }
    .footer-section { padding: 18px 28px 22px; border-radius: 0 0 30px 30px; }

    .question-text { text-align: center; color: #2F2142; font-size: 20px; margin: 14px 0 18px; font-weight: 600; letter-spacing: 0.15px; line-height: 1.4; }
    .progress-text { text-align: center; color: var(--ink-2); font-size: 13px; font-weight: 600; margin-top: 8px; letter-spacing: 0.4px; }
    .stProgress { background-color: rgba(155,127,181,.15) !important; border-radius: 12px; margin-bottom: 12px; height: 8px; }
    .stProgress > div > div > div > div { background: linear-gradient(90deg, var(--purple-1) 0%, var(--purple-2) 100%) !important; border-radius: 12px; height: 8px; }
    hr { border: none; margin: 0; display: none !important; }

    .stButton { animation: slideUp 0.5s ease-out; }
    .stButton > button {width: 100%; border-radius: 18px; padding: 12px 18px; font-size: 15px; font-weight: 600; margin-bottom: 8px;
        border: 3px solid var(--blue-1) !important; background: #FFF !important; color: #1F3F9F !important;
        letter-spacing: 0.4px; line-height: 1.5; transition: all .35s cubic-bezier(.4,0,.2,1);
        box-shadow: 0 6px 16px rgba(53,93,203,.25), 0 2px 8px rgba(0,0,0,.12) !important;}
    .stButton > button:hover {
        background: linear-gradient(135deg, var(--blue-1) 0%, var(--blue-2) 100%) !important; color: #FFF !important;
        transform: translateY(-4px); border-color: var(--blue-1) !important;
        box-shadow: 0 16px 45px rgba(53,93,203,.45), 0 8px 20px rgba(0,0,0,.2) !important;
    }
    .stButton > button:active { transform: translateY(-2px); }
    .analyze-btn { background: linear-gradient(135deg, var(--purple-1) 0%, var(--purple-2) 100%) !important; color: #FFF !important; border: 3px solid var(--purple-1) !important; font-size: 18px !important; padding: 20px 35px !important; }

    .login-signup-container { display: flex; gap: 20px; justify-content: center; margin-top: 25px; }
    .login-button, .signup-button { flex: 1; font-weight: 700 !important; }
    .login-button { border: 3px solid #8B7BA8 !important; color: #8B7BA8 !important; background: #FFF !important; }
    .signup-button { border: 3px solid var(--purple-1) !important; color: #FFF !important; background: linear-gradient(135deg, var(--purple-1) 0%, var(--purple-2) 100%) !important; }

    .stTextInput label { color: #2F2142 !important; font-weight: 600 !important; }
    .stTextInput input { color: #221532 !important; background: #FFF !important; border: 2px solid #6C4B92 !important; }
    .stTextInput input::placeholder { color: #6A5A82 !important; }

    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #F5F0FF 0%, #FAFBFF 100%);
        border-radius: 18px; padding: 16px; border: 2px solid #D4C5E2;
        box-shadow: 0 4px 12px rgba(155,127,181,.15);
    }
    [data-testid="stExpander"] {
        border: 2px solid #D4C5E2 !important; border-radius: 14px !important;
        background: linear-gradient(135deg, #FAFBFF 0%, #F5F8FF 100%) !important;
    }
    .card-style {
        background: linear-gradient(135deg, #FFF 0%, #FAFBFF 100%);
        border-radius: 16px; padding: 16px; border: 2px solid #E8DFF5; margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(155,127,181,.1), 0 2px 8px rgba(0,0,0,.05);
    }
    .footer-text { text-align: center; color: #7A6B8F; font-size: 15px; margin-bottom: 12px; font-weight: 500; }

    .content-placeholder {
        background: linear-gradient(135deg, rgba(229,216,241,.82) 0%, rgba(222,208,236,.8) 50%, rgba(216,199,232,.84) 100%);
        border-radius: 32px; position: relative; overflow: hidden; margin: 10px;
    }
    .content-placeholder::before {
        content: ''; position: absolute; inset: 0; pointer-events: none; z-index: 1;
        background: linear-gradient(45deg, transparent 25%, rgba(255,255,255,.12) 50%, transparent 75%);
        animation: shimmer 3.5s infinite;
    }
    .content-placeholder > * { position: relative; z-index: 2; }

    .left-section, .middle-section, .right-section { display: block; height: auto; min-height: 0; }
    .left-section { text-align: center; }
    .middle-section { padding: 22px 20px 26px; border-radius: 24px; min-width: 760px; max-width: 760px; margin: 0 auto; }
    .middle-panel { background: #FFF; border: none; padding: 0; box-shadow: none; }
    .right-section { background: transparent; border-radius: 20px; min-width: 300px; max-width: 300px; margin: 0 auto; }

    .good-thoughts-header { color: #2D1A42; font-size: 20px; font-weight: 700; text-align: center; margin: 18px 0 10px; letter-spacing: 0.3px; font-family: 'Poppins', sans-serif; }
    .good-thoughts-container { border-radius: 14px; padding: 8px 0; }
    .thought-item {
        color: #5B4B6F; font-size: 20px; font-weight: 700; text-align: center; line-height: 1.5;
        padding: 12px; margin: 14px 0; border-radius: 10px; transition: all .6s ease;
        background: rgba(155,127,181,.15); border: 1px solid rgba(155,127,181,.15);
    }
    .thought-item:hover { background: linear-gradient(135deg, rgba(155,127,181,.15) 0%, rgba(200,170,220,.15) 100%); border-color: rgba(155,127,181,.3); transform: translateY(-2px); }
    .thought-animated-box { min-height: 90px; display: flex; align-items: center; justify-content: center; }

    .left-section h1 { font-size: 46px; margin: 10px 0 6px; }
    .left-section h2 { font-size: 26px; font-weight: 600; margin: 0 0 12px; }
    .auth-header { color: var(--ink-1); font-size: 18px; font-weight: 700; text-align: center; margin: 0 0 10px; font-family: 'Poppins', sans-serif; }
    .auth-subtext { color: #4D3468; font-size: 12px; text-align: center; line-height: 1.5; font-weight: 600; letter-spacing: 0.2px; }

    @keyframes pulse { 0%,100% {opacity:1; transform:scale(1);} 50% {opacity:.8; transform:scale(1.06);} }
    .logo, .logo img { animation: pulse 2.5s cubic-bezier(0.4, 0, 0.6, 1) infinite; }
    @keyframes slideUp { from {opacity:0; transform:translateY(20px);} to {opacity:1; transform:translateY(0);} }
    @keyframes shimmer { 0% {transform:translateX(-150%);} 100% {transform:translateX(150%);} }

    @media (max-width: 1200px) {
        .left-section h1 { font-size: 44px; }
        .left-section h2 { font-size: 24px; }
        .thought-item { font-size: 17px; padding: 14px 12px; }
        .good-thoughts-header { font-size: 20px; }
        .middle-section, .right-section { min-width: auto; max-width: 100%; }
    }
    
</style>
""", unsafe_allow_html=True)

def get_static_models():
    """Return available trained model metadata (with safe fallback values)."""
    try:
        from pathlib import Path
        from scripts.integrated_model_inference import (
            build_or_load_kmeans_bundle,
            load_baseline_assets,
        )

        project_root = Path(APP_DIR)
        baseline_bundle, baseline_meta = load_baseline_assets(project_root)
        kmeans_bundle = build_or_load_kmeans_bundle(project_root)

        baseline_metrics = baseline_meta.get("metrics", {}) if isinstance(baseline_meta, dict) else {}
        baseline_accuracy = float(baseline_metrics.get("Accuracy", 0.0))
        baseline_macro_recall = float(baseline_metrics.get("Macro_Recall", 0.0))

        baseline_info = {
            "feature_cols": baseline_meta.get("features", []),
            "accuracy": baseline_accuracy,
            "f1_score": baseline_macro_recall,
            "class_names": baseline_meta.get("class_names", []),
            "model_type": baseline_meta.get("model_type", "Multinomial Logistic Regression"),
            "bundle_loaded": bool(baseline_bundle),
        }
        kmeans_info = {
            "feature_cols": kmeans_bundle.get("feature_cols", []),
            "n_clusters": int(getattr(kmeans_bundle.get("kmeans"), "n_clusters", 4)),
            "cluster_mapping": kmeans_bundle.get("cluster_to_burnout_class", {}),
        }

        return {
            "baseline_multinomial": baseline_info,
            "kmeans": kmeans_info,
            # Backward-compatible aliases used by earlier UI sections.
            "logistic_regression": baseline_info,
            "gradient_boosting": baseline_info,
        }
    except Exception:
        return {
            "baseline_multinomial": {
                "feature_cols": REQUIRED_FIELDS,
                "accuracy": 0.0,
                "f1_score": 0.0,
                "class_names": ["Very Low (Q1)", "Low (Q2)", "Moderate (Q3)", "High (Q4)"],
                "model_type": "Unavailable",
                "bundle_loaded": False,
            },
            "kmeans": {
                "feature_cols": ["Sleep_Quality", "Physical_Activity", "CGPA", "Social_Support", "Counseling", "Financial_Stress"],
                "n_clusters": 4,
                "cluster_mapping": {},
            },
            "logistic_regression": {
                "feature_cols": REQUIRED_FIELDS,
                "accuracy": 0.0,
                "f1_score": 0.0,
            },
            "gradient_boosting": {
                "feature_cols": REQUIRED_FIELDS,
                "accuracy": 0.0,
                "f1_score": 0.0,
            },
        }

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
        "Semester": "📅 how many semesters have you enrolled in?",
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
        "Semester": ["15", "16", "17", "18", "19","20", "21","22","23","24","25","26","27","28","29","30","30+"],
        "Residence_Type": ["Home", "Hostel", "Apartment", "Dorm", "Other"],
    }
    return options.get(field_name, [])


# Persistence layer (Supabase-only)
def get_required_supabase_client():
    """Return a ready Supabase client or raise when unavailable."""
    client = get_supabase_client()
    if client is None:
        raise RuntimeError("Supabase is not configured or currently unavailable.")
    return client

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
    """Upsert payload into Supabase and track sync errors in session state."""
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
                "Supabase host could not be resolved. Data operations are paused "
                "for this session until connectivity is restored."
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
    """Validate Supabase connectivity for required app tables."""
    try:
        client = get_required_supabase_client()
        client.table("users").select("user_id").limit(1).execute()
        st.session_state["last_supabase_sync_error"] = None
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"Supabase setup check failed: {exc}"

    # Authentication and profile data access
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

def create_user(first_name: str, last_name: str, email: str, password: str, phone_number: str = "", city: str = "", zip_code: str = "") -> tuple[bool, str]:
    """Create a user account in Supabase."""
    normalized_email = email.strip().lower()
    password_hash, password_salt = hash_password(password)
    user_id = str(uuid.uuid4())

    try:
        client = get_required_supabase_client()
        client.table("users").insert({"user_id": user_id, "first_name": first_name.strip(), "last_name": last_name.strip(), "email": normalized_email, "phone_number": phone_number.strip() or None, "city": city.strip() or None, "zip_code": zip_code.strip() or None, "password_hash": password_hash, "password_salt": password_salt, "is_verified": False}).execute()
        st.session_state["last_supabase_sync_error"] = None
        return True, user_id
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"users create failed: {exc}"
        if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
            return False, "A user with this email already exists."
        return False, "Unable to create account right now. Please try again."


def authenticate_user(email: str, password: str) -> tuple[bool, Optional[dict], str]:
    """Authenticate user by email and password."""
    normalized_email = email.strip().lower()
    try:
        client = get_required_supabase_client()
        response = (
            client.table("users")
            .select("user_id,first_name,last_name,email,password_hash,password_salt")
            .eq("email", normalized_email)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        row = rows[0] if rows else None
    except Exception:
        return False, None, "Login service is currently unavailable."

    if not row:
        return False, None, "No account found with this email."

    if not isinstance(row, dict):
        return False, None, "Login service returned unexpected data format."

    user_data = {
        "user_id": str(row.get("user_id") or ""),
        "first_name": str(row.get("first_name") or ""),
        "last_name": str(row.get("last_name") or ""),
        "email": str(row.get("email") or ""),
    }
    expected_hash = str(row.get("password_hash") or "")
    password_salt = str(row.get("password_salt") or "")
    if not verify_password(password, expected_hash, password_salt):
        return False, None, "Invalid password."

    return True, user_data, ""


def upsert_profile(user_id: str, age: Optional[int], lifestyle_parameters: str, personal_details: str) -> bool:
    """Create or update profile details for a user."""
    try:
        client = get_required_supabase_client()
        client.table("profile").upsert(
            {
                "user_id": user_id,
                "age": age,
                "lifestyle_parameters": {"text": lifestyle_parameters.strip()} if lifestyle_parameters.strip() else {},
                "personal_details": {"text": personal_details.strip()} if personal_details.strip() else {},
            },
            on_conflict="user_id",
        ).execute()
        st.session_state["last_supabase_sync_error"] = None
        return True
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"profile upsert failed: {exc}"
        return False


def get_user_static_answers(user_id: str) -> dict:
    """Return saved baseline answers (age/course/gender) for a user, if present."""
    try:
        client = get_required_supabase_client()
        response = (
            client.table("users")
            .select("survey_age,survey_course,survey_gender")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = response.data or []
        row = rows[0] if rows else None
    except Exception:
        return {}

    if not isinstance(row, dict):
        return {}

    return {"Age": row.get("survey_age"),"Course": row.get("survey_course"),"Gender": row.get("survey_gender"),"Relationship": row.get("survey_relationship"),"CGPA": row.get("survey_cgpa")}

def save_user_static_answer_if_missing(user_id: str, field_name: str, field_value: str) -> None:
    """Persist first submitted baseline answer only once for the user."""
    column_name = STATIC_USER_FIELDS.get(field_name)
    value = (field_value or "").strip()
    if not column_name or not value:
        return

    try:
        client = get_required_supabase_client()
        current_response = (
            client.table("users")
            .select(column_name)
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        rows = current_response.data or []
        row = rows[0] if rows else None
        current_value = row.get(column_name) if isinstance(row, dict) else ""
        if str(current_value).strip():
            return

        client.table("users").update({column_name: value}).eq("user_id", user_id).execute()
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"users static answer update failed: {exc}"


def save_user_daily_input_to_sql(user_id: str, answers: dict, prediction: dict, cluster: int) -> tuple[bool, str]:
    """Save one daily questionnaire response per user to Supabase."""
    today_value = date.today().isoformat()
    submitted_at = datetime.now().isoformat()

    try:
        client = get_required_supabase_client()
        client.table("daily_inputs").insert({"user_id": user_id, "input_date": today_value, "submitted_at": submitted_at, "answers_json": answers, "prediction_json": prediction, "cluster": cluster}).execute()
        st.session_state["last_data_save_error"] = None
        st.session_state["last_supabase_sync_error"] = None
        return True, ""
    except Exception as exc:
        error_text = str(exc).lower()
        if "duplicate" in error_text or "unique" in error_text:
            return False, "You have already submitted today's input. Please come back tomorrow."
        st.session_state["last_data_save_error"] = f"daily_inputs save failed: {exc}"
        st.session_state["last_supabase_sync_error"] = f"daily_inputs save failed: {exc}"
        return False, "Unable to save your daily input right now."

def has_user_submitted_today(user_id: str) -> bool:
    """Check whether user already submitted daily survey today."""
    today_value = date.today().isoformat()
    try:
        client = get_required_supabase_client()
        response = (
            client.table("daily_inputs")
            .select("entry_id")
            .eq("user_id", user_id)
            .eq("input_date", today_value)
            .limit(1)
            .execute()
        )
        return bool(response.data)
    except Exception:
        return False


def upsert_daily_feedback(user_id: str, input_date: str, recommendation_followed: Optional[bool], recommendation_helpful: Optional[bool], feedback_rating: Optional[int], app_feedback: str) -> tuple[bool, str]:
    """Create or update per-day recommendation/app feedback for a user."""
    try:
        client = get_required_supabase_client()
        client.table("daily_feedback").upsert(
            {
                "user_id": user_id,
                "input_date": input_date,
                "recommendation_followed": None if recommendation_followed is None else int(recommendation_followed),
                "recommendation_helpful": None if recommendation_helpful is None else int(recommendation_helpful),
                "feedback_rating": feedback_rating,
                "app_feedback": app_feedback.strip() or None,
            },
            on_conflict="user_id,input_date",
        ).execute()
        st.session_state["last_supabase_sync_error"] = None
        return True, ""
    except Exception as exc:
        st.session_state["last_supabase_sync_error"] = f"daily_feedback save failed: {exc}"
        return False, "Unable to save feedback right now."


def get_user_daily_history(user_id: str) -> pd.DataFrame:
    """Fetch a user's day-by-day survey history with optional feedback."""
    try:
        client = get_required_supabase_client()
        daily_response = (
            client.table("daily_inputs")
            .select("entry_id,user_id,input_date,submitted_at,prediction_json,cluster")
            .eq("user_id", user_id)
            .order("input_date", desc=False)
            .execute()
        )
        feedback_response = (
            client.table("daily_feedback")
            .select("user_id,input_date,recommendation_followed,recommendation_helpful,feedback_rating,app_feedback")
            .eq("user_id", user_id)
            .execute()
        )
    except Exception:
        return pd.DataFrame()

    history_df = pd.DataFrame(daily_response.data or [])

    if history_df.empty:
        return history_df

    feedback_df = pd.DataFrame(feedback_response.data or [])
    if not feedback_df.empty:
        history_df = history_df.merge(
            feedback_df,
            on=["user_id", "input_date"],
            how="left",
        )
    else:
        history_df["recommendation_followed"] = None
        history_df["recommendation_helpful"] = None
        history_df["feedback_rating"] = None
        history_df["app_feedback"] = None

    parsed_predictions = history_df["prediction_json"].apply(parse_prediction_json)
    history_df["stress_level"] = parsed_predictions.apply(lambda p: p.get("stress_level"))
    history_df["anxiety_score"] = parsed_predictions.apply(lambda p: p.get("anxiety_score"))
    history_df["depression_score"] = parsed_predictions.apply(lambda p: p.get("depression_score"))
    history_df["mental_health_pct"] = parsed_predictions.apply(lambda p: p.get("mental_health_pct"))
    return history_df


def parse_prediction_json(prediction_json: Any) -> dict:
    """Safely parse prediction JSON payload."""
    try:
        if not prediction_json:
            return {}
        if isinstance(prediction_json, dict):
            return prediction_json
        parsed = json.loads(prediction_json)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def get_admin_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch users and daily input records for admin view."""
    try:
        client = get_required_supabase_client()
        users_response = (
            client.table("users")
            .select("user_id,first_name,last_name,email,phone_number,city,zip_code,created_at")
            .order("created_at", desc=True)
            .execute()
        )
        daily_response = (
            client.table("daily_inputs")
            .select("entry_id,user_id,input_date,submitted_at,prediction_json,cluster")
            .order("submitted_at", desc=True)
            .execute()
        )
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

    users_df = pd.DataFrame(users_response.data or [])
    daily_df = pd.DataFrame(daily_response.data or [])

    if users_df.empty:
        users_df = pd.DataFrame(columns=[
            "user_id", "first_name", "last_name", "email", "phone_number", "city", "zip_code", "created_at", "total_entries", "last_input_date"
        ])
    if daily_df.empty:
        daily_df = pd.DataFrame(columns=["entry_id", "user_id", "input_date", "submitted_at", "prediction_json", "cluster"])
    else:
        aggregates = (
            daily_df.groupby("user_id", as_index=False)
            .agg(total_entries=("entry_id", "count"), last_input_date=("input_date", "max"))
        )
        users_df = users_df.merge(aggregates, on="user_id", how="left")

    users_df["total_entries"] = users_df["total_entries"].fillna(0).astype(int)

    if not daily_df.empty:
        parsed_predictions = daily_df["prediction_json"].apply(parse_prediction_json)
        daily_df["stress_level"] = parsed_predictions.apply(lambda p: p.get("stress_level"))
        daily_df["anxiety_score"] = parsed_predictions.apply(lambda p: p.get("anxiety_score"))
        daily_df["depression_score"] = parsed_predictions.apply(lambda p: p.get("depression_score"))

    return users_df, daily_df

# UI renderers and app navigation
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
        users_display = users_display.rename(columns={"first_name": "First Name", "last_name": "Last Name", "email": "Email", "phone_number": "Phone", "city": "City", "zip_code": "ZIP", "total_entries": "Entries", "last_input_date": "Last Input Date"})
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
        "last_data_save_error": None,
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
                try:
                    from pathlib import Path
                    from scripts.integrated_model_inference import integrated_predict

                    integrated_output = integrated_predict(answers, Path(APP_DIR))
                    baseline_output = integrated_output.get("baseline_multinomial", {})
                    kmeans_output = integrated_output.get("unsupervised_kmeans", {})
                    cluster = int(kmeans_output.get("cluster", 0))

                    prediction = {
                        "baseline_multinomial": baseline_output,
                        "unsupervised_kmeans": kmeans_output,
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
                except Exception as exc:
                    st.warning(f"⚠️ Unable to run baseline + KMeans inference: {exc}")

        if st.button("🔄 Start New Assessment", key="reset_assessment_btn", width="stretch"):
            reset_survey_state()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Results Display
        if st.session_state.get("show_results"):
            st.markdown("<div class='results-section'>", unsafe_allow_html=True)
            
            prediction = st.session_state.get("last_prediction")
            
            st.markdown("<h3 style='text-align: center;'>📊 Baseline + KMeans Output</h3>", unsafe_allow_html=True)
            baseline_output = prediction.get("baseline_multinomial", {}) if isinstance(prediction, dict) else {}
            kmeans_output = prediction.get("unsupervised_kmeans", {}) if isinstance(prediction, dict) else {}

            col_model_1, col_model_2 = st.columns(2)
            with col_model_1:
                st.markdown("#### Baseline Model")
                st.json(baseline_output)
            with col_model_2:
                st.markdown("#### KMeans Model")
                st.json(kmeans_output)
            
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
            st.markdown("<div class='auth-subtext'>Supabase history enabled</div>", unsafe_allow_html=True)
            if st.session_state.get("last_data_save_error"):
                st.caption(f"Data save note: {st.session_state.get('last_data_save_error')}")
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
