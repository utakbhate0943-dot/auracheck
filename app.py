import os
from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client

from training_module.fetch_supabase_data import fetch_students_mental_health
from training_module.gradient_boosting_model import train_gradient_boosting_from_csv
from training_module.kmeans_model import train_kmeans_from_csv
from training_module.logistic_regression_model import train_logistic_regression_from_csv
from training_module.model_training import build_input_row, build_preprocessor, build_wellbeing_target

load_dotenv()

st.set_page_config(page_title="AuraCheck", page_icon="💜", layout="wide")


def get_supabase() -> Client | None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


def get_openai_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def initialize_state():
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def behavior_remark(cluster_id: int) -> str:
    mapping = {
        0: "Stable pattern: maintain routine and sleep discipline.",
        1: "Moderate strain pattern: improve recovery habits and social support.",
        2: "High pressure pattern: prioritize stress reduction and structured breaks.",
    }
    return mapping.get(cluster_id, "Mixed behavior pattern: monitor stress and maintain healthy habits.")


def analyze_with_openai(client: OpenAI | None, remark: str, stress_level: int, wellbeing_pct: float) -> str:
    if client is None:
        return (
            "Try three small steps today: 10 minutes breathing, 20 minutes walk, and one supportive conversation. "
            "If stress continues to rise, consider talking to a counselor."
        )
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    try:
        response = client.responses.create(
            model=model_name,
            input=[
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
        return response.output_text
    except Exception:
        return (
            "OpenAI recommendations are temporarily unavailable due to API quota/configuration. "
            "For now: sleep 7-8h, daily movement, reduce caffeine late evening, and schedule one calming break every 3 hours."
        )


def render_auth_section(supabase: Client | None):
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


def save_prediction_log(supabase: Client | None, remark: str, recommendation: str):
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


def render_page_one(df: pd.DataFrame, model_bundle: dict, grad_bundle: dict, kmeans_bundle: dict, supabase: Client | None, ai_client: OpenAI | None):
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
            st.plotly_chart(gauge_stress, use_container_width=True)
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
            st.plotly_chart(gauge_health, use_container_width=True)

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


def render_page_two(supabase: Client | None):
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
        st.line_chart(chart_df[selected_cols], use_container_width=True)


def main():
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
        ["Page 1 • Chat + Login/Signup", "Page 2 • User Profile Dashboard"],
    )

    if page == "Page 1 • Chat + Login/Signup":
        render_page_one(df, model_bundle, grad_bundle, kmeans_bundle, supabase, ai_client)
    else:
        render_page_two(supabase)


if __name__ == "__main__":
    main()
