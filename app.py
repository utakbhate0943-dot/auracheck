import os
import json
from datetime import date, datetime
from typing import Dict, Any, Optional
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
        background: linear-gradient(135deg, #E8DFF5 0%, #F0E8FA 50%, #E6D9F0 100%);
        min-height: 100vh;
        padding: 6px 15px;
    }

    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
    }
    
    /* Main container card - large single card */
    .main {
        background: linear-gradient(135deg, rgba(232, 223, 245, 0.6) 0%, rgba(240, 232, 250, 0.6) 50%, rgba(230, 217, 240, 0.6) 100%);
        border-radius: 40px;
        padding: 0;
        box-shadow: 0 35px 100px rgba(124, 91, 166, 0.3), 0 15px 50px rgba(155, 127, 181, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.7);
        margin: 25px auto;
        max-width: 750px;
        width: 100%;
        border: 5px solid #9B7FB5;
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
        border: 3px solid #5B7FEA !important;
        background-color: #FFFFFF !important;
        color: #5B7FEA !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        margin-bottom: 14px;
        box-shadow: 0 6px 16px rgba(91, 127, 234, 0.2), 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        font-family: 'Inter', sans-serif;
        letter-spacing: 0.4px;
        text-transform: none;
        line-height: 1.5;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5B7FEA 0%, #4A6FD9 100%) !important;
        color: #FFFFFF !important;
        transform: translateY(-4px);
        border-color: #5B7FEA !important;
        box-shadow: 0 16px 45px rgba(91, 127, 234, 0.5), 0 8px 20px rgba(0, 0, 0, 0.15) !important;
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
        color: #9B7FB5;
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
        color: #5B4B6F;
        font-family: 'Inter', sans-serif;
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 15px;
    }
    
    /* Question text */
    .question-text {
        text-align: center;
        color: #5B4B6F;
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
        color: #9B7FB5;
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
        color: #6B5B7F;
        font-weight: 500;
        line-height: 1.6;
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
        background: linear-gradient(135deg, rgba(200, 170, 220, 0.25) 0%, rgba(190, 160, 210, 0.22) 50%, rgba(200, 170, 220, 0.28) 100%);
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
        color: #3F2456;
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
        color: #9B7FB5;
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


def save_user_response_to_json(answers: dict, prediction: dict, cluster: int) -> None:
    """Save user response to JSON file."""
    try:
        response_data = {
            "timestamp": datetime.now().isoformat(),
            "user_inputs": answers,
            "predictions": prediction,
            "cluster": cluster
        }
        
        json_file_path = "Dataset/user_responses.json"
        all_responses = []
        
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r') as f:
                try:
                    all_responses = json.load(f)
                except:
                    all_responses = []
        
        all_responses.append(response_data)
        
        with open(json_file_path, 'w') as f:
            json.dump(all_responses, f, indent=2)
        
    except Exception as e:
        pass


def initialize_state() -> None:
    """Initialize session state."""
    defaults = {
        "last_answers": {},
        "last_prediction": None,
        "last_cluster": None,
        "show_results": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main():
    """Main Streamlit Application."""
    
    models = get_static_models()
    initialize_state()
    
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
        
        positive_thoughts = [
            "🌟 You are capable of overcoming challenges",
            "💚 Your mental health matters and deserves attention",
            "🌈 Every day is a fresh opportunity for growth",
            "💫 You have strength within you",
            "🌸 Self-care is not selfish, it's essential",
            "⭐ Progress over perfection always",
            "🎯 Your feelings are valid and important",
            "🌊 Challenges help you grow stronger",
            "💡 You deserve to be happy and healthy",
            "🦋 Transformation starts with self-compassion"
        ]

        thoughts_js = json.dumps(positive_thoughts)
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
        
        required_fields = [
            "Age", "Course", "Gender", "CGPA", "Sleep_Quality", 
            "Physical_Activity", "Diet_Quality", "Social_Support", 
            "Relationship", "Substance_Use", "Counseling", 
            "Family_History", "Chronic_Illness", "Financial_Stress", 
            "Extracurricular", "Semester", "Residence_Type"
        ]
        
        current_question_idx = len(st.session_state.get("last_answers", {}))
        answers = st.session_state.get("last_answers", {})
        
        # Progress Section
        if current_question_idx > 0:
            progress_pct = current_question_idx / len(required_fields)
            st.progress(progress_pct)
            st.markdown(f"<p class='progress-text'>Question {current_question_idx} of {len(required_fields)}</p>", unsafe_allow_html=True)
        
        # Questions Display
        st.markdown("<div class='questions-section'>", unsafe_allow_html=True)
        
        if current_question_idx < len(required_fields):
            current_field = required_fields[current_question_idx]
            options = get_field_options(current_field)
            question = get_question_for_field(current_field)
            
            st.markdown(f"<div class='question-text'>{question}</div>", unsafe_allow_html=True)
            
            if options:
                for idx, option in enumerate(options):
                    if st.button(option, key=f"btn_{current_field}_{idx}", use_container_width=True):
                        clean_value = option.split("(")[0].strip() if "(" in option else option
                        clean_value = ''.join(c for c in clean_value if ord(c) < 0x1F600 or ord(c) > 0x1F64F)
                        clean_value = clean_value.strip()
                        answers[current_field] = clean_value if clean_value else option
                        st.session_state["last_answers"] = answers
                        st.rerun()

                skip_col_left, skip_col_mid, skip_col_right = st.columns([1, 1.2, 1])
                with skip_col_mid:
                    if st.button("⏭️ Skip This Question", key=f"skip_{current_field}", use_container_width=True):
                        answers[current_field] = "Skipped"
                        st.session_state["last_answers"] = answers
                        st.rerun()
            else:
                user_input = st.text_input(f"Your answer:", key=f"input_{current_field}")
                if user_input:
                    answers[current_field] = user_input
                    st.session_state["last_answers"] = answers
                    st.rerun()
        else:
            st.markdown("<div style='text-align: center; padding: 30px 0;'><h3 style='color: #3F2456;'>✨ All set!</h3><p style='color: #7A6B8F;'>Click below to analyze your wellbeing</p></div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Analyze Button
        st.markdown("<div class='analyze-section'>", unsafe_allow_html=True)
        if st.button("🔍 Analyze My Results", key="analyze_btn", use_container_width=True):
            if len(answers) < len(required_fields):
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
                st.plotly_chart(fig_stress, use_container_width=True)
            
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
                st.plotly_chart(fig_wellbeing, use_container_width=True)
            
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

        st.markdown("</div>", unsafe_allow_html=True)
        
    
    # ========== RIGHT COLUMN: Authentication ==========
    with right_col:
        
        st.markdown("<div class='auth-header'>Join AuraCheck</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
        
        st.button("⭐ Sign Up", key="signup_btn", use_container_width=True)
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
        st.button("📝 Log In", key="login_btn", use_container_width=True)
        
        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
        st.markdown("<div class='auth-subtext'>Save your progress &<br/>track your journey</div>", unsafe_allow_html=True)
    
    # --- CLOSE CONTENT PLACEHOLDER ---
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
