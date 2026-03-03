import os
from datetime import date, datetime
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client
import shap

from training_module.fetch_supabase_data import fetch_students_mental_health
from training_module.gradient_boosting_model import train_gradient_boosting_from_csv
from training_module.kmeans_model import train_kmeans_from_csv
from training_module.logistic_regression_model import train_logistic_regression_from_csv
from training_module.model_training import build_input_row, build_preprocessor, build_wellbeing_target
from training_module.seasonal_model import train_seasonal_from_logs, get_seasonal_insight, forecast_stress_trend

load_dotenv()

st.set_page_config(page_title="AuraCheck", page_icon="💜", layout="wide")


def get_supabase() -> Optional[Client]:
    """
    Initialize and return Supabase client if credentials available.
    
    Returns None if SUPABASE_URL or SUPABASE_ANON_KEY not configured.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def get_openai_client() -> Optional[OpenAI]:
    """
    Initialize and return OpenAI client if API key available.
    
    Returns None if OPENAI_API_KEY not configured.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def initialize_state() -> None:
    """
    Initialize Streamlit session state with default values.
    
    Handles resets for chat history, predictions, user auth, cluster assignments,
    and seasonal analysis across page reloads.
    """
    defaults = {
        "messages": [
            {
                "role": "assistant",
                "content": "Welcome to AuraCheck 💜. Complete the questionnaire for analysis and recommendations.",
            }
        ],
        "chat_history": [],
        "auth_user": None,
        "last_prediction": None,
        "last_cluster": None,
        "last_answers": {},
        "last_auto_reco": None,
        "seasonal_bundle": None,
        "shap_values": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def behavior_remark(cluster_id: int) -> str:
    """
    Generate tailored behavioral advice based on KMeans cluster assignment.
    
    Args:
        cluster_id: Cluster ID (0-2) from unsupervised clustering
    
    Returns:
        str: Human-readable behavior insight and recommendation
    
    Design Rationale:
        Cluster 0 (Stable): High resilience, reinforce good habits
        Cluster 1 (Moderate): Needs support improvement, social connection focus
        Cluster 2 (High Pressure): Immediate stress reduction, structured breaks
    """
    mapping = {
        0: "Stable pattern: maintain routine and sleep discipline.",
        1: "Moderate strain pattern: improve recovery habits and social support.",
        2: "High pressure pattern: prioritize stress reduction and structured breaks.",
    }
    return mapping.get(cluster_id, "Mixed behavior pattern: monitor stress and maintain healthy habits.")


def analyze_with_openai(client: Optional[OpenAI], remark: str, stress_level: int, wellbeing_pct: float) -> str:
    """
    Generate personalized wellness recommendations via OpenAI API.
    
    If OpenAI unavailable or rate-limited, returns safe fallback recommendations
    based on stress level and wellbeing percentage.
    
    Args:
        client: OpenAI client instance or None
        remark: User's context/concern description
        stress_level: Predicted stress (0-5)
        wellbeing_pct: Mental health percentage (0-100)
    
    Returns:
        str: AI-generated wellness recommendations or fallback advice
    
    Design Rationale:
    - Graceful fallback prevents user experience degradation when API unavailable
    - Reuses business context (stress level, wellbeing) in prompt for personalization
    - Non-diagnostic language complies with healthcare guidelines
    """
    if client is None:
        return (
            "Try three small steps today: 10 minutes breathing, 20 minutes walk, and one supportive conversation. "
            "If stress continues to rise, consider talking to a counselor."
        )
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        response = client.chat.completions.create(
            model=model_name,
            max_tokens=200,
            messages=[
                {
                    "role": "system",
                    "content": "You are a supportive wellness assistant. Be practical, concise, and non-diagnostic.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Predicted stress level: {stress_level}/5. "
                        f"Predicted mental health: {wellbeing_pct:.1f}%. "
                        f"User context: {remark}. "
                        "Give short analysis and 5 actionable recommendations."
                    ),
                },
            ],
        )
        return response.choices[0].message.content
    except Exception:
        return (
            "OpenAI recommendations are temporarily unavailable due to API quota/configuration. "
            "For now: sleep 7-8h, daily movement, reduce caffeine late evening, and schedule one calming break every 3 hours."
        )


def get_shap_explanation(model: Any, preprocessor: Any, feature_names: list, user_input_row: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate SHAP (SHapley Additive exPlanations) values for model prediction.
    
    **Explainability Design:**
    SHAP values show each feature's contribution to the prediction, answering:
    "Why is my stress level 4/5 instead of average 2.5/5?"
    
    Args:
        model: Trained Pipeline with preprocessing + estimator
        preprocessor: ColumnTransformer from training pipeline
        feature_names: List of original feature column names
        user_input_row: Single-row DataFrame with user questionnaire responses
    
    Returns:
        Dict with keys:
            - 'top_features': List of (feature_name, impact_direction, magnitude) tuples
            - 'explanation': Readable description of top 3 contributors
            - 'values': Raw SHAP values array
            - 'features': Preprocessed feature names
    
    Design Rationale:
    - SHAP uses game theory to assign feature importance in an interpretable way
    - TreeExplainer fast for tree-based models; KernelExplainer for linear
    - Shows both positive (increase stress) and negative (reduce stress) contributors
    - Top 3 features communicated to user in simple language
    """
    try:
        # Extract preprocessed features
        X_preprocessed = preprocessor.transform(user_input_row)
        
        # Try to get feature names from preprocessor
        try:
            preprocessor_feature_names = preprocessor.get_feature_names_out()
        except Exception:
            preprocessor_feature_names = [f"Feature {i}" for i in range(X_preprocessed.shape[1])]
        
        # Extract the actual model (after preprocessing in pipeline)
        if hasattr(model, 'named_steps'):
            ensemble_model = model.named_steps.get('model', model)
        else:
            ensemble_model = model
        
        # Calculate SHAP values
        if hasattr(ensemble_model, 'predict_proba'):
            explainer = shap.TreeExplainer(ensemble_model) if hasattr(ensemble_model, 'estimators_') else shap.KernelExplainer(ensemble_model.predict, X_preprocessed)
        else:
            explainer = shap.KernelExplainer(ensemble_model.predict, X_preprocessed)
        
        shap_values = explainer.shap_values(X_preprocessed)
        
        # Handle multi-class case (take class with highest impact)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # Use first class as example
        
        # Get top 3 contributing features
        shap_importance = np.abs(shap_values[0])
        top_indices = np.argsort(shap_importance)[-3:][::-1]
        
        top_features = []
        for idx in top_indices:
            feat_name = preprocessor_feature_names[idx] if idx < len(preprocessor_feature_names) else f"Feature {idx}"
            impact = "increases" if shap_values[0][idx] > 0 else "decreases"
            magnitude = abs(shap_values[0][idx])
            top_features.append((feat_name, impact, magnitude))
        
        explanation = _format_shap_explanation(top_features)
        
        return {
            'top_features': top_features,
            'explanation': explanation,
            'values': shap_values,
            'features': list(preprocessor_feature_names),
        }
    except Exception as e:
        return {
            'top_features': [],
            'explanation': f"SHAP analysis unavailable: {str(e)[:50]}",
            'values': None,
            'features': feature_names,
        }


def _format_shap_explanation(top_features: list) -> str:
    """Format SHAP feature contributions into readable explanation."""
    if not top_features:
        return "Unable to calculate feature contributions."
    
    lines = ["**Top factors affecting your stress prediction:**"]
    for i, (feat, direction, mag) in enumerate(top_features, 1):
        # Simplify feature name
        clean_name = feat.replace("_", " ").title()
        lines.append(f"{i}. {clean_name} {direction} stress (impact: {mag:.2f})")
    
    return "\n".join(lines)


def render_auth_section(supabase: Optional[Client]) -> None:
    """
    Render Supabase email magic-link authentication section.
    
    Allows users to sign up or login via email OTP/magic link for profile
    and dashboard access on Page 2.
    
    Args:
        supabase: Supabase client instance or None
    """
    st.subheader("Signup / Login with Email OTP / Magic Link")
    if supabase is None:
        st.info("Set SUPABASE_URL and SUPABASE_ANON_KEY to enable email login.")
        return

    if st.session_state["auth_user"]:
        current_user = st.session_state["auth_user"]
        st.success(f"Logged in as {current_user.get('email', 'user')}")
        if st.button("Logout"):
            st.session_state["auth_user"] = None
            st.rerun()
        return

    email = st.text_input("Email", placeholder="your@email.com")
    if st.button("Send Magic Link") and email:
        try:
            supabase.auth.sign_in_with_otp({"email": email})
            st.success("Magic link sent. Check your email to complete login.")
        except Exception as exc:
            st.error(f"Could not send magic link: {exc}")


def save_prediction_log(supabase: Optional[Client], remark: str, recommendation: str) -> None:
    """
    Save user prediction and recommendation to database daily_logs table.
    
    Persists predictions after user completes assessment for historical tracking
    and trend analysis on Page 2 dashboard.
    
    Args:
        supabase: Supabase client or None
        remark: User's text context or concern
        recommendation: Generated recommendation text
    
    Returns None silently if saving fails (gracefully degrades).
    """
    user = st.session_state.get("auth_user")
    prediction = st.session_state.get("last_prediction")
    if supabase is None or user is None or prediction is None:
        return

    payload = {
        "user_id": user["id"],
        "log_date": str(date.today()),
        "predicted_stress_level": int(prediction.get("stress_level_logistic", 0)),
        "predicted_mental_health_pct": float(prediction.get("mental_health_pct", 0)),
        "remark": remark,
        "recommendation": recommendation,
    }
    try:
        supabase.table("daily_logs").insert(payload).execute()
    except Exception:
        pass


def render_page_one(df: pd.DataFrame, model_bundle: Dict[str, Any], grad_bundle: Dict[str, Any], 
                   kmeans_bundle: Dict[str, Any], supabase: Optional[Client], ai_client: Optional[OpenAI]) -> None:
    """
    Render Page 1: Assessment Questionnaire + Predictions + Chat.
    
    Features:
    - Questionnaire form with dynamic field population from dataset
    - Dual-model stress prediction (Logistic Regression + Gradient Boosting)
    - SHAP explainability for feature contributions
    - KMeans behavioral clustering with personalized remarks
    - OpenAI personalized recommendations
    - Chat interface for follow-up questions
    - Email magic-link signup/login
    
    Args:
        df: Training dataset for form options and feature scaling
        model_bundle: Logistic Regression model result dict
        grad_bundle: Gradient Boosting model result dict
        kmeans_bundle: KMeans clustering model result dict
        supabase: Supabase client or None
        ai_client: OpenAI client or None
    """
    st.markdown("""
        <style>
        body, .stApp { background: #f6f3fa !important; }
        .aura-title { font-size: 2.5rem; font-weight: 700; color: #4b2aad; margin-bottom: 0.3rem; }
        .aura-caption { color: #7a6bb0; font-size: 1.1rem; margin-bottom: 1.2rem; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="aura-title">AuraCheck 💜</div>', unsafe_allow_html=True)
    st.markdown('<div class="aura-caption">Your private mental health assistant. Assessment & Chat with Aura.</div>', unsafe_allow_html=True)

    col_chat, col_form = st.columns([1, 2], gap="small")

    with col_chat:
        st.markdown("### Chat with AuraBot")
        user_input = st.text_input("You:", key="chat_input")
        if st.button("Send", key="send_btn") and user_input:
            st.session_state["chat_history"].append(("user", user_input))
            prediction = st.session_state.get("last_prediction")
            stress_ctx = prediction.get("stress_level_logistic", 2) if prediction else 2
            health_ctx = prediction.get("mental_health_pct", 55.0) if prediction else 55.0
            bot_reply = analyze_with_openai(ai_client, user_input, stress_ctx, health_ctx)
            st.session_state["chat_history"].append(("bot", bot_reply))
            save_prediction_log(supabase, user_input, bot_reply)

        for sender, msg in st.session_state["chat_history"][-8:]:
            if sender == "user":
                st.markdown(f"**You:** {msg}")
            else:
                st.markdown(f"**AuraBot:** {msg}")

    with col_form:
        st.markdown("### Quick Assessment Questionnaire")
        if df is None or df.empty:
            st.warning("No data available for assessment form.")
            return

        with st.form(key="assessment_form_main"):
            c1, c2 = st.columns(2)
            answers = {}
            with c1:
                if "Age" in df.columns:
                    answers["Age"] = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()))
                if "Gender" in df.columns:
                    answers["Gender"] = st.selectbox("Gender", sorted(df["Gender"].dropna().unique().tolist()))
                if "Course" in df.columns:
                    answers["Course"] = st.selectbox("Course", sorted(df["Course"].dropna().unique().tolist()))
                if "CGPA" in df.columns:
                    answers["CGPA"] = st.slider("CGPA", float(df["CGPA"].min()), float(df["CGPA"].max()), float(df["CGPA"].median()), 0.01)
                if "Sleep_Quality" in df.columns:
                    answers["Sleep_Quality"] = st.selectbox("Sleep Quality", sorted(df["Sleep_Quality"].dropna().unique().tolist()))
                if "Physical_Activity" in df.columns:
                    answers["Physical_Activity"] = st.selectbox("Physical Activity", sorted(df["Physical_Activity"].dropna().unique().tolist()))
            with c2:
                if "Diet_Quality" in df.columns:
                    answers["Diet_Quality"] = st.selectbox("Diet Quality", sorted(df["Diet_Quality"].dropna().unique().tolist()))
                if "Social_Support" in df.columns:
                    answers["Social_Support"] = st.selectbox("Social Support", sorted(df["Social_Support"].dropna().unique().tolist()))
                if "Financial_Stress" in df.columns:
                    answers["Financial_Stress"] = st.slider("Financial Stress", int(df["Financial_Stress"].min()), int(df["Financial_Stress"].max()), int(df["Financial_Stress"].median()))
                if "Extracurricular_Involvement" in df.columns:
                    answers["Extracurricular_Involvement"] = st.selectbox("Extracurricular Involvement", sorted(df["Extracurricular_Involvement"].dropna().unique().tolist()))
                if "Semester_Credit_Load" in df.columns:
                    answers["Semester_Credit_Load"] = st.slider("Semester Credit Load", int(df["Semester_Credit_Load"].min()), int(df["Semester_Credit_Load"].max()), int(df["Semester_Credit_Load"].median()))
                if "Residence_Type" in df.columns:
                    answers["Residence_Type"] = st.selectbox("Residence Type", sorted(df["Residence_Type"].dropna().unique().tolist()))

            submitted = st.form_submit_button("Predict stress and mental health")

        if submitted:
            x_log = build_input_row(model_bundle["feature_cols"], answers, df)
            stress_log = int(np.clip(np.round(model_bundle["model"].predict(x_log)[0]), 0, 5))

            x_grad = build_input_row(grad_bundle["feature_cols"], answers, df)
            stress_grad = int(np.clip(np.round(grad_bundle["model"].predict(x_grad)[0]), 0, 5))

            wellbeing_pct = float(build_wellbeing_target(df).mean())

            x_kmeans_ref = df[kmeans_bundle["feature_cols"]].copy()
            preproc = build_preprocessor(x_kmeans_ref)
            preproc.fit(x_kmeans_ref)
            x_kmeans = build_input_row(kmeans_bundle["feature_cols"], answers, df)
            cluster = int(kmeans_bundle["model"].predict(preproc.transform(x_kmeans))[0])

            st.session_state["last_prediction"] = {
                "stress_level_logistic": stress_log,
                "stress_level_gradient": stress_grad,
                "mental_health_pct": wellbeing_pct,
            }
            st.session_state["last_cluster"] = cluster
            st.session_state["last_answers"] = answers
            st.session_state["last_auto_reco"] = analyze_with_openai(
                ai_client,
                "Based on my questionnaire results, what should I do now to reduce stress and feel lighter?",
                stress_log,
                wellbeing_pct,
            )

        prediction = st.session_state.get("last_prediction")
        cluster = st.session_state.get("last_cluster")
        if prediction is None or cluster is None:
            return

        card1, card2, card3 = st.columns(3)
        card1.metric("Stress Level", f"{prediction['stress_level_logistic']} / 5")
        card2.metric("Mental Health", f"{prediction['mental_health_pct']:.1f}%")
        card3.metric("Behaviour Remark", behavior_remark(cluster))

        g1, g2 = st.columns(2)
        with g1:
            gauge_stress = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction["stress_level_logistic"],
                title={"text": "Stress Level (Gauge)"},
                gauge={
                    "axis": {"range": [0, 5]},
                    "bar": {"color": "#a259e6"},
                    "steps": [
                        {"range": [0, 2], "color": "#b7e4c7"},
                        {"range": [2, 4], "color": "#ffe066"},
                        {"range": [4, 5], "color": "#ff6f69"},
                    ],
                },
            ))
            st.plotly_chart(gauge_stress, width='stretch')
        with g2:
            gauge_health = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction["mental_health_pct"],
                title={"text": "Mental Health % (Gauge)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#43aa8b"},
                    "steps": [
                        {"range": [0, 50], "color": "#ff6f69"},
                        {"range": [50, 75], "color": "#ffe066"},
                        {"range": [75, 100], "color": "#b7e4c7"},
                    ],
                },
            ))
            st.plotly_chart(gauge_health, width='stretch')

        # Model Performance Metrics
        st.markdown("#### Model Performance & Reliability")
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("LR Accuracy", f"{model_bundle['accuracy']:.1%}")
        metrics_cols[1].metric("LR F1 Score", f"{model_bundle['f1_score']:.1%}")
        metrics_cols[2].metric("GB Accuracy", f"{grad_bundle['accuracy']:.1%}")
        metrics_cols[3].metric("GB F1 Score", f"{grad_bundle['f1_score']:.1%}")
        st.caption("LR=Logistic Regression (interpretable), GB=Gradient Boosting (robust). Metrics from 80% test data.")

        # SHAP Explainability
        st.markdown("#### What Influenced Your Stress Prediction?")
        x_input = build_input_row(model_bundle["feature_cols"], st.session_state.get("last_answers", {}), df)
        shap_result = get_shap_explanation(
            model_bundle["model"],
            build_preprocessor(df[model_bundle["feature_cols"]]),
            model_bundle["feature_cols"],
            x_input
        )
        st.markdown(shap_result['explanation'])

        st.info(f"Gradient Boosting Stress Level: {prediction['stress_level_gradient']}")
        st.info(f"KMeans Behaviour Group: Group {cluster + 1} • {behavior_remark(cluster)}")

        st.markdown("**AI Recommendations to deal with your situation:**")
        st.write(st.session_state.get("last_auto_reco") or "Recommendations will appear after prediction.")

        user_remark = st.text_area(
            "Share your thoughts or concerns for personalized recommendations:",
            key="openai_remark",
            height=80,
        )
        if st.button("Get Personalized Recommendations", key="openai_btn") and user_remark:
            personal_reco = analyze_with_openai(
                ai_client,
                user_remark,
                prediction["stress_level_logistic"],
                prediction["mental_health_pct"],
            )
            st.markdown("**AI Recommendations (Personalized):**")
            st.write(personal_reco)
            save_prediction_log(supabase, user_remark, personal_reco)

        st.markdown("---")
        st.markdown(
            "**If you like our results, sign up using your email address.** "
            "You will get an OTP or magic link on email. "
            "After signup, open Page 2 to track daily progress and dashboards."
        )
        render_auth_section(supabase)


def render_page_two(supabase: Optional[Client]) -> None:
    """
    Render Page 2: User Profile + Daily Tracker Dashboard.
    
    Features:
    - Profile form (name, age, gender, course, wellness goals)
    - Daily check-in tracker (mood score, sleep hours, note)
    - Trend dashboard (stress, mental health, mood, sleep by date)
    - Seasonal analysis showing stress trends and forecasts
    - Historical data visualization using Plotly
    
    Args:
        supabase: Supabase client or None
    """
    st.title("User Profile & Mental Health Tracker 💜")
    st.caption("Page 2: Profile + Daily Tracker Dashboard")

    user = st.session_state.get("auth_user")
    if user is None:
        st.warning("Please login on Page 1 to access profile and dashboard.")
        return
    if supabase is None:
        st.error("Supabase is not configured.")
        return

    user_id = user["id"]

    profile_data = None
    try:
        result = supabase.table("profiles").select("*").eq("user_id", user_id).limit(1).execute()
        if result.data:
            profile_data = result.data[0]
    except Exception:
        profile_data = None

    def pval(key: str, default):
        if isinstance(profile_data, dict):
            return profile_data.get(key, default)
        return default

    st.subheader("Profile")
    with st.form("profile_form"):
        full_name = st.text_input("Full name", value=pval("full_name", ""))
        age = st.number_input("Age", min_value=10, max_value=100, value=int(pval("age", 22)))
        gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male", "Female", "Other"].index(pval("gender", "Male")) if pval("gender", "Male") in ["Male", "Female", "Other"] else 0)
        course = st.text_input("Course", value=pval("course", ""))
        goals = st.text_area("Wellness goals", value=pval("goals", ""))
        save_profile = st.form_submit_button("Save profile")

    if save_profile:
        payload = {
            "user_id": user_id,
            "phone": user.get("phone"),
            "full_name": full_name,
            "age": age,
            "gender": gender,
            "course": course,
            "goals": goals,
            "updated_at": datetime.utcnow().isoformat(),
        }
        try:
            supabase.table("profiles").upsert(payload).execute()
            st.success("Profile saved.")
        except Exception as exc:
            st.error(f"Could not save profile: {exc}")

    st.subheader("Daily tracker check-in")
    with st.form("daily_tracker"):
        mood = st.slider("Mood score", 1, 10, 6)
        sleep_hours = st.slider("Sleep hours", 0.0, 12.0, 7.0, 0.5)
        note = st.text_area("Today note")
        submit_log = st.form_submit_button("Save daily check-in")

    if submit_log:
        payload = {
            "user_id": user_id,
            "log_date": str(date.today()),
            "mood_score": mood,
            "sleep_hours": sleep_hours,
            "remark": note,
        }
        try:
            supabase.table("daily_logs").insert(payload).execute()
            st.success("Daily check-in saved.")
        except Exception as exc:
            st.error(f"Could not save daily check-in: {exc}")

    st.subheader("Improvement dashboard")
    try:
        logs_result = (
            supabase.table("daily_logs")
            .select("log_date,predicted_stress_level,predicted_mental_health_pct,mood_score,sleep_hours")
            .eq("user_id", user_id)
            .order("log_date", desc=False)
            .execute()
        )
        logs = pd.DataFrame(logs_result.data)
    except Exception:
        logs = pd.DataFrame()

    if logs.empty:
        st.info("No tracker data yet. Add entries to see dashboard trends.")
        return

    for col in ["predicted_stress_level", "predicted_mental_health_pct", "mood_score", "sleep_hours"]:
        if col in logs.columns:
            logs[col] = pd.to_numeric(logs[col], errors="coerce")

    mcols = st.columns(4)
    mcols[0].metric("Latest Stress", f"{logs['predicted_stress_level'].dropna().iloc[-1] if logs['predicted_stress_level'].dropna().shape[0] else '-'}")
    mcols[1].metric("Latest Mental Health %", f"{logs['predicted_mental_health_pct'].dropna().iloc[-1]:.1f}%" if logs['predicted_mental_health_pct'].dropna().shape[0] else "-")
    mcols[2].metric("Latest Mood", f"{logs['mood_score'].dropna().iloc[-1] if logs['mood_score'].dropna().shape[0] else '-'}")
    mcols[3].metric("Latest Sleep (hrs)", f"{logs['sleep_hours'].dropna().iloc[-1]:.1f}" if logs['sleep_hours'].dropna().shape[0] else "-")

    chart_df = logs.set_index("log_date")
    selected_cols = [c for c in ["predicted_mental_health_pct", "predicted_stress_level", "mood_score", "sleep_hours"] if c in chart_df.columns]
    if selected_cols:
        st.line_chart(chart_df[selected_cols], width='stretch')
    
    # Seasonal stress analysis
    st.markdown("#### Seasonal & Temporal Trends")
    logs_for_seasonal = logs[["log_date", "predicted_stress_level"]].copy()
    logs_for_seasonal["log_date"] = pd.to_datetime(logs_for_seasonal["log_date"])
    seasonal_bundle = train_seasonal_from_logs(logs_for_seasonal)
    
    seasonal_insight = get_seasonal_insight(seasonal_bundle)
    st.info(seasonal_insight)
    
    # 4-week forecast if enough data
    if seasonal_bundle.get('model') is not None:
        forecast_df = forecast_stress_trend(seasonal_bundle, weeks=4)
        if forecast_df is not None and not forecast_df.empty:
            st.markdown("**4-Week Stress Level Forecast:**")
            forecast_plot = go.Figure()
            forecast_plot.add_trace(go.Scatter(
                x=forecast_df['ds'], 
                y=forecast_df['yhat'], 
                mode='lines', 
                name='Forecast',
                line=dict(color='#a259e6')
            ))
            forecast_plot.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_upper'],
                fill=None,
                mode='lines',
                name='Upper bound',
                line=dict(width=0),
                showlegend=False
            ))
            forecast_plot.add_trace(go.Scatter(
                x=forecast_df['ds'],
                y=forecast_df['yhat_lower'],
                fill='tonexty',
                mode='lines',
                name='Lower bound',
                line=dict(width=0),
                showlegend=False
            ))
            forecast_plot.update_layout(height=300, title="Predicted Stress Trend (4 weeks)")
            st.plotly_chart(forecast_plot, width='stretch')


def render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    """
    Render confusion matrix as heatmap visualization.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        title: Title for the heatmap
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=[f"Pred {i}" for i in range(cm.shape[1])],
        y=[f"True {i}" for i in range(cm.shape[0])],
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Predicted Stress Level",
        yaxis_title="Actual Stress Level",
        height=400,
    )
    st.plotly_chart(fig, width='stretch')


def render_page_three(model_bundle: Dict[str, Any], grad_bundle: Dict[str, Any], 
                      kmeans_bundle: Dict[str, Any]) -> None:
    """
    Render Page 3: ML Model Performance Metrics & KPIs.
    
    Displays comprehensive model evaluation metrics including:
    - Classification metrics (accuracy, precision, recall, F1)
    - Clustering metrics (silhouette, Davies-Bouldin)
    - Error metrics (MSE, MAE)
    - Confusion matrices
    - Model comparison
    
    Args:
        model_bundle: Logistic Regression model result dict
        grad_bundle: Gradient Boosting model result dict
        kmeans_bundle: KMeans clustering model result dict
    """
    st.title("📊 ML Model Performance Metrics & KPIs")
    st.caption("Page 3: Comprehensive model evaluation across all algorithms")
    
    # Overview metrics in cards
    st.markdown("## 🎯 Overall Model Performance Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Logistic Regression Accuracy",
            value=f"{model_bundle['accuracy']:.1%}",
            delta=f"F1: {model_bundle['f1_score']:.1%}"
        )
    
    with col2:
        st.metric(
            label="Gradient Boosting Accuracy",
            value=f"{grad_bundle['accuracy']:.1%}",
            delta=f"F1: {grad_bundle['f1_score']:.1%}"
        )
    
    with col3:
        st.metric(
            label="KMeans Silhouette Score",
            value=f"{kmeans_bundle['silhouette_score']:.3f}",
            delta="(range: -1 to 1)"
        )
    
    # Detailed comparison table
    st.markdown("## 📈 Detailed Model Metrics Comparison")
    
    metrics_data = {
        "Metric": [
            "Accuracy",
            "Precision (weighted)",
            "Recall (weighted)",
            "F1 Score (weighted)",
            "Mean Squared Error",
            "Mean Absolute Error",
            "Test Sample Size"
        ],
        "Logistic Regression": [
            f"{model_bundle['accuracy']:.4f}",
            f"{model_bundle['precision']:.4f}",
            f"{model_bundle['recall']:.4f}",
            f"{model_bundle['f1_score']:.4f}",
            f"{model_bundle['mse']:.4f}",
            f"{model_bundle['mae']:.4f}",
            str(model_bundle['test_size'])
        ],
        "Gradient Boosting": [
            f"{grad_bundle['accuracy']:.4f}",
            f"{grad_bundle['precision']:.4f}",
            f"{grad_bundle['recall']:.4f}",
            f"{grad_bundle['f1_score']:.4f}",
            f"{grad_bundle['mse']:.4f}",
            f"{grad_bundle['mae']:.4f}",
            str(grad_bundle['test_size'])
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, width='stretch', hide_index=True)
    
    # Classification Models Details
    st.markdown("## 🔍 Classification Models Deep Dive")
    
    tab1, tab2 = st.tabs(["Logistic Regression", "Gradient Boosting"])
    
    with tab1:
        st.markdown("### Logistic Regression - Interpretability & Speed")
        st.markdown("""
        **Model Strengths:**
        - Fast inference (suitable for real-time predictions)
        - Interpretable coefficients show feature importance
        - Probabilistic output for confidence scores
        - Ideal for linear relationships
        """)
        
        col_lr1, col_lr2 = st.columns(2)
        with col_lr1:
            st.markdown("**Key Performance Indicators (KPIs):**")
            st.metric("Accuracy", f"{model_bundle['accuracy']:.2%}", help="Percentage of correct predictions")
            st.metric("Precision", f"{model_bundle['precision']:.2%}", help="Positive predictions that were correct")
            st.metric("Recall", f"{model_bundle['recall']:.2%}", help="Actual positives correctly identified")
        
        with col_lr2:
            st.markdown("**Error Metrics:**")
            st.metric("MSE", f"{model_bundle['mse']:.4f}", help="Mean Squared Error")
            st.metric("MAE", f"{model_bundle['mae']:.4f}", help="Mean Absolute Error")
            st.metric("F1 Score", f"{model_bundle['f1_score']:.4f}", help="Harmonic mean of precision & recall")
        
        # Confusion matrix for LR
        st.markdown("### Confusion Matrix - Logistic Regression")
        render_confusion_matrix(model_bundle['y_test'], model_bundle['y_pred'], 
                               "Logistic Regression Confusion Matrix")
        
        # Interpretation guide
        st.markdown("""
        **How to Read the Confusion Matrix:**
        - **Diagonal**: Correct predictions (true positives/negatives)
        - **Off-diagonal**: Errors (false positives/negatives)
        - **Better model**: Larger diagonal, smaller off-diagonal values
        """)
    
    with tab2:
        st.markdown("### Gradient Boosting - Robustness & Generalization")
        st.markdown("""
        **Model Strengths:**
        - Captures non-linear relationships
        - Ensemble method reduces overfitting
        - Handles feature interactions automatically
        - Often outperforms linear models
        """)
        
        col_gb1, col_gb2 = st.columns(2)
        with col_gb1:
            st.markdown("**Key Performance Indicators (KPIs):**")
            st.metric("Accuracy", f"{grad_bundle['accuracy']:.2%}", help="Percentage of correct predictions")
            st.metric("Precision", f"{grad_bundle['precision']:.2%}", help="Positive predictions that were correct")
            st.metric("Recall", f"{grad_bundle['recall']:.2%}", help="Actual positives correctly identified")
        
        with col_gb2:
            st.markdown("**Error Metrics:**")
            st.metric("MSE", f"{grad_bundle['mse']:.4f}", help="Mean Squared Error")
            st.metric("MAE", f"{grad_bundle['mae']:.4f}", help="Mean Absolute Error")
            st.metric("F1 Score", f"{grad_bundle['f1_score']:.4f}", help="Harmonic mean of precision & recall")
        
        # Confusion matrix for GB
        st.markdown("### Confusion Matrix - Gradient Boosting")
        render_confusion_matrix(grad_bundle['y_test'], grad_bundle['y_pred'], 
                               "Gradient Boosting Confusion Matrix")
        
        st.markdown("""
        **Model Comparison:**
        Compare confusion matrices to see which model is better at:
        - Correctly identifying stressed students
        - Avoiding false negatives (missed cases)
        - Balancing precision vs recall
        """)
    
    # Clustering Model Details
    st.markdown("## 🎲 Behavioral Clustering Model (KMeans)")
    
    col_km1, col_km2, col_km3 = st.columns(3)
    
    with col_km1:
        st.metric(
            "Silhouette Score",
            f"{kmeans_bundle['silhouette_score']:.3f}",
            help="Higher is better (-1 to 1)"
        )
    
    with col_km2:
        st.metric(
            "Davies-Bouldin Index",
            f"{kmeans_bundle['davies_bouldin_index']:.3f}",
            help="Lower is better (cluster separation)"
        )
    
    with col_km3:
        st.metric(
            "Inertia",
            f"{kmeans_bundle['inertia']:.2f}",
            help="Sum of squared distances to centers"
        )
    
    st.markdown("""
    **KMeans Clustering Interpretation:**
    - **Silhouette Score (-1 to 1)**: Measures how well-separated clusters are
      - >0.5: Good cluster separation
      - 0.3-0.5: Reasonable clusters
      - <0.3: Possible overlap between clusters
    
    - **Davies-Bouldin Index**: Average similarity between each cluster and its most similar cluster
      - Lower values indicate better clustering
      - Typically 0-5 range
    
    - **Inertia**: Sum of squared distances to nearest cluster center
      - Lower values indicate tighter, more compact clusters
      - Decreases as k increases (not comparable across different k)
    
    **Applied Usage:**
    The model groups students into 3 behavioral patterns:
    - Cluster 0: "Stable pattern" - good stress management
    - Cluster 1: "Moderate strain" - manageable stress levels
    - Cluster 2: "High pressure" - significant stress detected
    """)
    
    # KPI Interpretation Guide
    st.markdown("## 📚 KPI Interpretation Guide")
    
    with st.expander("Accuracy - What does it mean?"):
        st.markdown("""
        **Accuracy = (TP + TN) / (TP + TN + FP + FN)**
        
        - Percentage of all predictions that were correct
        - Range: 0% to 100%
        - **Good accuracy**: >85% for stress prediction
        - **Limitation**: Can be misleading if classes are imbalanced
        
        **Example:** 85% accuracy means 85 out of 100 predictions were correct
        """)
    
    with st.expander("Precision - What does it mean?"):
        st.markdown("""
        **Precision = TP / (TP + FP)**
        
        - Of all positive predictions, how many were actually correct?
        - Range: 0% to 100%
        - **High precision**: Few false alarms (low false positive rate)
        - **Use when**: False positives are costly
        
        **Example:** 80% precision means if model predicts stress, it's correct 80% of the time
        """)
    
    with st.expander("Recall - What does it mean?"):
        st.markdown("""
        **Recall = TP / (TP + FN)**
        
        - Of all actual positives, how many did we catch?
        - Range: 0% to 100%
        - **High recall**: Few missed cases (low false negative rate)
        - **Use when**: False negatives are costly (missing stressed students is bad)
        
        **Example:** 85% recall means we identify 85% of actually stressed students
        """)
    
    with st.expander("F1 Score - What does it mean?"):
        st.markdown("""
        **F1 = 2 × (Precision × Recall) / (Precision + Recall)**
        
        - Harmonic mean of precision and recall
        - Range: 0 to 1 (multiply by 100 for percentage)
        - **Balanced metric**: Considers both precision and recall
        - **Use when**: Need to balance false positives and false negatives
        
        **Better than accuracy when**: Classes are imbalanced
        """)
    
    with st.expander("MSE & MAE - What do they mean?"):
        st.markdown("""
        **Mean Squared Error (MSE)** = Average of squared differences
        - Penalizes large errors more heavily
        - Range: 0 to infinity (lower is better)
        - Units: Square of target variable units
        
        **Mean Absolute Error (MAE)** = Average absolute difference
        - Linear error: all errors penalized equally
        - Range: 0 to infinity (lower is better)
        - Same units as target variable (more interpretable)
        
        **Example for stress (1-5 scale):**
        - MAE = 0.5 means predictions off by 0.5 levels on average
        - MSE would flag predictions off by 2+ levels as especially bad
        """)
    
    # Model Selection Guide
    st.markdown("## 🎓 Which Model Should You Use?")
    
    comparison_data = {
        "Aspect": [
            "🎯 Accuracy",
            "📊 Interpretability",
            "⚡ Speed",
            "🧠 Complexity",
            "📈 Non-linear",
            "🎯 Best For"
        ],
        "Logistic Regression": [
            f"{model_bundle['accuracy']:.1%}",
            "Excellent (coefficients)",
            "Very fast",
            "Simple",
            "No",
            "Baseline, interpretability"
        ],
        "Gradient Boosting": [
            f"{grad_bundle['accuracy']:.1%}",
            "Good (feature importance)",
            "Moderate",
            "Complex",
            "Yes",
            "Production, accuracy"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, width='stretch', hide_index=True)
    
    st.markdown("""
    **Recommendation:**
    - Use **Logistic Regression** for initial screening and understanding which factors affect stress
    - Use **Gradient Boosting** for production system requiring highest accuracy
    - Ensemble both predictions by averaging for maximum robustness
    """)


def main() -> None:
    """
    Main application entry point orchestrating all components.
    
    **Application Flow:**
    1. Initialize session state for persistence across page reloads
    2. Fetch live training dataset from Supabase
    3. Train ML models:
       - Logistic Regression (interpretable stress classification)
       - Gradient Boosting (robust ensemble prediction)
       - KMeans (unsupervised behavioral clustering)
    4. Initialize external clients:
       - Supabase (auth + database)
       - OpenAI (personalized recommendations)
    5. Render appropriate page based on sidebar selection
    
    **Page 1: Assessment + Analysis + Chat**
    - Dynamic questionnaire form
    - Dual-model stress prediction
    - SHAP explainability showing feature contributions
    - KMeans behavior grouping
    - OpenAI recommendations with fallback
    - Email magic-link sign up
    - Chat interface for follow-up
    
    **Page 2: Profile + Dashboard**
    - User profile management
    - Daily check-in tracker
    - Trend visualization (stress, mood, sleep, mental health %)
    - Seasonal analysis showing stress patterns
    - 4-week stress forecast using Prophet
    
    **Error Handling:**
    - Graceful degradation if external services unavailable
    - Safe fallback recommendations if OpenAI API down
    - Clear error messages for missing configuration
    """
    initialize_state()

    csv_path = fetch_students_mental_health()
    if not csv_path:
        st.error("No data found in Supabase table students_mental_health.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        st.error(f"Could not read dataset: {exc}")
        return

    try:
        model_bundle = train_logistic_regression_from_csv(csv_path)
        grad_bundle = train_gradient_boosting_from_csv(csv_path)
        kmeans_bundle = train_kmeans_from_csv(csv_path)
    except Exception as exc:
        st.error(f"Model setup failed due to dataset columns/schema: {exc}")
        return

    supabase = get_supabase()
    ai_client = get_openai_client()

    page = st.sidebar.radio(
        "Navigation",
        ["Page 1 • Chat + Login/Signup", "Page 2 • User Profile Dashboard", "Page 3 • Model Metrics & KPIs"],
    )

    if page == "Page 1 • Chat + Login/Signup":
        render_page_one(df, model_bundle, grad_bundle, kmeans_bundle, supabase, ai_client)
    elif page == "Page 2 • User Profile Dashboard":
        render_page_two(supabase)
    else:
        render_page_three(model_bundle, grad_bundle, kmeans_bundle)


if __name__ == "__main__":
    main()
