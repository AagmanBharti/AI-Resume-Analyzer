import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.preprocess import clean_text
from src.skills import extract_skills, skill_gap
from src.scoring import hybrid_match_score, resume_quality, section_check
from utils.pdf_reader import extract_pdf_text
from src.report import generate_report

st.set_page_config(
    page_title="Enterprise AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
label_encoder = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

st.title("🚀 Enterprise AI Resume Analyzer")

mode = st.sidebar.selectbox("Mode", ["Candidate Mode", "Recruiter Mode"])
job_desc = st.text_area("Paste Job Description")

# ================= Candidate Mode =================

if mode == "Candidate Mode":
    uploaded = st.file_uploader("Upload Resume PDF", type=["pdf"])

    if uploaded and job_desc:

        resume_text = extract_pdf_text(uploaded)

        resume_text_clean = clean_text(resume_text)
        job_desc_clean = clean_text(job_desc)

        # ---------- Category Prediction ----------
        vec = vectorizer.transform([resume_text_clean])
        pred = model.predict(vec)
        category = label_encoder.inverse_transform(pred)[0]

        # ---------- Skill Extraction ----------
        resume_skills = extract_skills(resume_text_clean)
        jd_skills = extract_skills(job_desc_clean)
        missing = skill_gap(resume_skills, jd_skills)

        # ---------- Hybrid Match Score ----------
        match, tfidf_part, semantic_part = hybrid_match_score(
            resume_text_clean,
            job_desc_clean,
            vectorizer
        )

        # ---------- Resume Quality ----------
        quality = resume_quality(match, len(resume_skills), resume_text_clean)

        # ---------- Section Check ----------
        sections = section_check(resume_text_clean)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Category", category)
            st.metric("Hybrid Match Score", f"{match}%")
            st.metric("Resume Quality Score", f"{quality}/100")

        with col2:
            st.subheader("Extracted Skills")
            st.write(resume_skills)

            st.subheader("Missing Skills")
            st.write(missing)

        # ================= Detailed Score Breakdown =================
        st.subheader("📊 Matching Breakdown")

        score_df = pd.DataFrame({
            "Type": ["Keyword (TF-IDF)", "Semantic (Embeddings)"],
            "Score": [tfidf_part, semantic_part]
        })

        st.dataframe(score_df, use_container_width=True)

        # ================= VISUAL MATCH SCORE BAR =================
        st.subheader("📊 Match Score Visualization")

        fig = plt.figure()
        plt.bar(["Hybrid Score"], [match])
        plt.ylim(0, 100)
        plt.ylabel("Score (%)")
        plt.title("Resume vs Job Description Match")

        st.pyplot(fig)

        # ================= ATS BREAKDOWN =================
        st.subheader("📌 ATS Scoring Breakdown")

        skill_component = min(len(resume_skills) * 3, 30)
        length_component = 20 if 400 <= len(resume_text_clean.split()) <= 800 else 10
        match_component = match * 0.5

        breakdown_df = pd.DataFrame({
            "Component": [
                "Job Match (50%)",
                "Skill Strength (30%)",
                "Resume Structure (20%)"
            ],
            "Score Contribution": [
                round(match_component, 2),
                skill_component,
                length_component
            ]
        })

        st.dataframe(breakdown_df, use_container_width=True)

        # ================= SECTION CHECK =================
        st.subheader("📑 Section Analysis")

        for sec, present in sections.items():
            st.write(f"{sec}: {'✅' if present else '❌'}")

        # ================= GENERATE REPORT =================
        report_path = "resume_evaluation_report.pdf"

        generate_report(report_path, {
            "Category": category,
            "Hybrid Match Score": f"{match}%",
            "Keyword Match": f"{tfidf_part}%",
            "Semantic Match": f"{semantic_part}%",
            "Quality Score": quality,
            "Skills": ", ".join(resume_skills),
            "Missing Skills": ", ".join(missing)
        })

        with open(report_path, "rb") as f:
            st.download_button(
                "Download Evaluation Report",
                f,
                file_name="resume_evaluation_report.pdf"
            )

# ================= Recruiter Mode =================

if mode == "Recruiter Mode":

    uploads = st.file_uploader(
        "Upload Multiple Resumes",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploads and job_desc:

        results = []
        job_desc_clean = clean_text(job_desc)

        for file in uploads:
            text = extract_pdf_text(file)
            clean = clean_text(text)

            score, _, _ = hybrid_match_score(
                clean,
                job_desc_clean,
                vectorizer
            )

            results.append((file.name, score))

        results = sorted(results, key=lambda x: x[1], reverse=True)

        df = pd.DataFrame(results, columns=["Resume", "Hybrid Match Score"])

        st.subheader("📊 Ranked Candidates")
        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download Ranking CSV",
            df.to_csv(index=False),
            file_name="ranking.csv"
        )
