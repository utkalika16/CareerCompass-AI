import json
from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ml_engine import (
    FEATURES, CAREER_PROFILES, CAREER_ACTIONS,
    generate_demo_dataset, evaluate_models,
    readiness_score, readiness_label, career_matches,
    skill_gaps, feature_importance, predict_student
)
from report_generator import build_pdf

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "careercompass_demo.csv"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="CareerCompass AI",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Styling ----------
st.markdown("""
<style>
.main {background: #f7f9fc;}
.block-container {padding-top: 1.2rem;}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(135deg, #102a43, #1f4e79);
    color: white;
    margin-bottom: 22px;
}
.hero h1 {font-size: 42px; margin-bottom: 6px;}
.hero p {font-size: 17px; opacity: .92;}
.card {
    padding: 18px;
    border-radius: 16px;
    background: white;
    border: 1px solid #e6eaf0;
    box-shadow: 0 5px 20px rgba(20,40,80,.05);
}
.small-note {color:#5c677d; font-size:13px;}
</style>
""", unsafe_allow_html=True)

# ---------- Cached training ----------
@st.cache_resource(show_spinner=False)
def load_system():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(exist_ok=True)
        generate_demo_dataset(1200, 42).to_csv(DATA_PATH, index=False)

    data = pd.read_csv(DATA_PATH)

    # Always evaluate on the same documented demo split.
    trained, metrics = evaluate_models(data)
    best_model = max(metrics, key=lambda name: metrics[name]["f1"])
    return data, trained, metrics, best_model

data, trained_models, metrics, best_model_name = load_system()
best_model = trained_models[best_model_name]

# ---------- Header ----------
st.markdown("""
<div class="hero">
<h1>🧭 CareerCompass AI</h1>
<p>Student Success & Placement Intelligence System</p>
<p>Placement prediction • Readiness assessment • Skill-gap analysis • Career recommendation</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Prototype data notice: the supplied notebook uses a synthetic/demo dataset. "
    "Use this app to demonstrate the integrated workflow; replace the demo data with real, "
    "properly labelled institutional data before making real placement decisions."
)

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "Student Assessment", "Career Explorer", "Model Evaluation", "Batch CSV", "About"]
)

st.sidebar.markdown("---")
st.sidebar.metric("Demo students", len(data))
st.sidebar.metric("Best model", best_model_name)
st.sidebar.caption("Model selection is based on F1 score on the demo test split.")

# ---------- Dashboard ----------
if page == "Dashboard":
    st.subheader("System overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", f"{len(data):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("ML models", len(trained_models))
    c4.metric("Careers", len(CAREER_PROFILES))

    st.markdown("### What the system answers")
    a, b, c = st.columns(3)
    with a:
        st.markdown("**1. Placement prediction**")
        st.write("Estimates whether the student is likely to meet the placement target learned from the training data.")
    with b:
        st.markdown("**2. Readiness + skill gaps**")
        st.write("Calculates a 0–100 readiness score and identifies weaker competencies for the selected career.")
    with c:
        st.markdown("**3. Career direction**")
        st.write("Ranks six project-defined career profiles and provides the Top-3 matches.")

    st.markdown("### Demo dataset distribution")
    dist = data["Placement"].map({0:"Not Placed", 1:"Placed"}).value_counts()
    st.bar_chart(dist)

    st.markdown("### Workflow")
    st.code(
        "Student Profile → Preprocessing → ML Models → Placement Prediction → "
        "Readiness → Career Matching → Skill Gaps → Improvement Roadmap",
        language="text"
    )

# ---------- Student assessment ----------
elif page == "Student Assessment":
    st.subheader("🎓 Student Assessment")
    st.write("Enter a student's current profile. The system produces one integrated assessment.")

    with st.form("student_form"):
        left, right = st.columns(2)

        with left:
            age = st.number_input("Age", 18, 40, 21)
            cgpa = st.number_input("CGPA", 0.0, 10.0, 8.0, 0.01)
            attendance = st.number_input("Attendance (%)", 0.0, 100.0, 85.0, 0.5)
            technical = st.slider("Technical Skill", 0, 100, 70)
            communication = st.slider("Communication Skill", 0, 100, 70)
            aptitude = st.slider("Aptitude Score", 0, 100, 70)
            coding = st.slider("Coding Score", 0, 100, 70)

        with right:
            projects = st.number_input("Projects", 0, 20, 2)
            internships = st.number_input("Internships", 0, 10, 1)
            certifications = st.number_input("Certifications", 0, 20, 2)
            extracurricular = st.number_input("Extracurricular Activities", 0, 20, 2)
            soft = st.slider("Soft Skill", 0, 100, 70)
            resume = st.slider("Resume Score", 0, 100, 70)
            interview = st.slider("Interview Score", 0, 100, 70)

        selected_model = st.selectbox(
            "Prediction model",
            list(trained_models.keys()),
            index=list(trained_models.keys()).index(best_model_name)
        )
        submitted = st.form_submit_button("🔎 Analyze Student", use_container_width=True)

    if submitted:
        student = {
            "Age": age, "CGPA": cgpa, "Attendance": attendance,
            "Technical_Skill": technical, "Communication_Skill": communication,
            "Aptitude_Score": aptitude, "Projects": projects,
            "Internships": internships, "Certifications": certifications,
            "Extracurricular": extracurricular, "Coding_Score": coding,
            "Soft_Skill": soft, "Resume_Score": resume, "Interview_Score": interview
        }

        model = trained_models[selected_model]
        pred, probability = predict_student(model, student)
        ready = readiness_score(student)
        ready_text = readiness_label(ready)
        careers = career_matches(student)
        top_career = careers.iloc[0]["Career"]
        gaps = skill_gaps(student, top_career)

        st.session_state["last_result"] = {
            "student": student, "model_name": selected_model,
            "prediction": pred, "probability": probability,
            "readiness": ready, "readiness_text": ready_text,
            "careers": careers, "gaps": gaps
        }

    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        st.markdown("### Assessment result")

        c1, c2, c3 = st.columns(3)
        c1.metric("Placement prediction", "Likely to be placed" if r["prediction"] else "Needs improvement")
        c2.metric("Placement probability", f"{r['probability']:.1f}%")
        c3.metric("Readiness score", f"{r['readiness']:.1f}%")

        st.progress(int(r["readiness"]))
        st.success(r["readiness_text"])

        st.markdown("### Top career recommendations")
        st.dataframe(r["careers"].head(3), use_container_width=True, hide_index=True)

        st.markdown("### Skill-gap analysis")
        if len(r["gaps"]):
            st.dataframe(r["gaps"], use_container_width=True, hide_index=True)
        else:
            st.success("No major gap detected for the top recommended career.")

        top = r["careers"].iloc[0]["Career"]
        st.markdown(f"### Personalized roadmap — {top}")
        for item in CAREER_ACTIONS[top]:
            st.checkbox(item, key=f"roadmap_{item}")

        pdf = build_pdf(
            r["student"], r["model_name"], r["prediction"], r["probability"],
            r["readiness"], r["readiness_text"], r["careers"], r["gaps"],
            CAREER_ACTIONS[top]
        )
        st.download_button(
            "📄 Download Student Report (PDF)",
            data=pdf,
            file_name="CareerCompass_Student_Report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

# ---------- Career explorer ----------
elif page == "Career Explorer":
    st.subheader("🧭 Career Explorer")
    st.write("Compare the six project-defined career profiles and understand what each path emphasizes.")

    selected = st.selectbox("Select career", list(CAREER_PROFILES.keys()))
    req = CAREER_PROFILES[selected]

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f"### {selected}")
        st.dataframe(
            pd.DataFrame(
                [{"Requirement": k.replace("_", " "), "Target": v} for k, v in req.items()]
            ),
            hide_index=True,
            use_container_width=True
        )
    with c2:
        st.markdown("### Suggested development actions")
        for item in CAREER_ACTIONS[selected]:
            st.write("• " + item)

    st.markdown("### All career profiles")
    rows = []
    for career, req in CAREER_PROFILES.items():
        rows.append({"Career": career, **req})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------- Model evaluation ----------
elif page == "Model Evaluation":
    st.subheader("📊 Model Evaluation")
    st.write("All six models from the project notebook are trained and evaluated using the same 80/20 stratified split.")

    metric_df = pd.DataFrame(metrics).T.sort_values("f1", ascending=False)
    st.dataframe(metric_df, use_container_width=True)

    st.markdown("### F1 comparison")
    st.bar_chart(metric_df["f1"])

    st.markdown("### Accuracy comparison")
    st.bar_chart(metric_df["accuracy"])

    st.markdown("### Feature importance / coefficient magnitude")
    imp = feature_importance(best_model)
    st.dataframe(imp.head(10), use_container_width=True, hide_index=True)
    st.bar_chart(imp.head(10).set_index("Feature")["Importance"])

    st.caption(
        "These metrics describe performance on the supplied synthetic/demo dataset. "
        "They are not evidence of real-world placement accuracy."
    )

# ---------- Batch CSV ----------
elif page == "Batch CSV":
    st.subheader("📁 Batch Student Analysis")
    st.write("Upload a CSV with the 14 input features. The app predicts placement and readiness for every row.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error("Missing columns: " + ", ".join(missing))
        else:
            clean = df[FEATURES].copy()
            for col in FEATURES:
                clean[col] = pd.to_numeric(clean[col], errors="coerce")
            if clean.isna().any().any():
                st.error("The CSV contains missing/non-numeric values in required features.")
            else:
                model = best_model
                preds = model.predict(clean)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(clean)[:, 1] * 100
                else:
                    probs = preds * 100

                output = df.copy()
                output["Placement_Prediction"] = preds
                output["Placement_Probability"] = np.round(probs, 2)
                output["Readiness_Score"] = [
                    round(readiness_score(row.to_dict()), 2)
                    for _, row in clean.iterrows()
                ]

                st.dataframe(output, use_container_width=True, hide_index=True)
                csv = output.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download predictions CSV",
                    csv,
                    "careercompass_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )

    st.markdown("### Download demo CSV")
    demo_csv = data[FEATURES + ["Placement"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download sample dataset",
        demo_csv,
        "careercompass_demo.csv",
        "text/csv",
        use_container_width=True
    )

# ---------- About ----------
else:
    st.subheader("ℹ️ About CareerCompass AI")
    st.markdown("""
### Project objective

CareerCompass AI is designed as an integrated Student Success and Placement Intelligence System.
The project combines student performance analysis, placement/employability prediction,
placement-readiness assessment, skill-gap identification and personalized career recommendation.

### Project workflow

**Data Collection → Preprocessing → Feature Engineering → Model Training → Evaluation →**
**Placement Prediction → Skill-Gap Analysis → Career Recommendation**

### Candidate models

- Logistic Regression
- Decision Tree
- Random Forest
- SVM
- KNN
- Neural Network

### Why this prototype is useful

Traditional systems often answer only one question: placement prediction or career recommendation.
This prototype combines both with a readiness score and an actionable skill-gap layer.

### Research-grounded scope

The supplied project documents identify academic performance, technical skills, soft skills,
communication, projects, internships, certifications, extracurricular activities and preferences
as relevant dimensions, and identify the integration of prediction, readiness, skill gaps and career
recommendation as the central project contribution.

### Limitation

The supplied notebook uses synthetic/demo data. Replace it with a properly collected and labelled
institutional dataset before treating the system as a real placement-decision system.
""")

st.markdown("---")
st.caption("CareerCompass AI • Student Success & Placement Intelligence System")
