import streamlit as st
import joblib
import pandas as pd
import re
from sklearn.base import BaseEstimator, TransformerMixin

# ------------------------
# 🧼 Custom Text Cleaner
# ------------------------
class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self

    def transform(self, X):
        return pd.Series(X).apply(self.clean_text)

    def clean_text(self, text):
        text = re.sub(r"\n", " ", str(text))
        text = re.sub(r"[^\w\s]", "", text)
        return text.lower()

# ------------------------
# 📦 Load Trained Models
# ------------------------
majors_model = joblib.load("majors_model.pkl")
majors_mlb = joblib.load("majors_mlb.pkl")

tech_model = joblib.load("technical_skills_model.pkl")
tech_mlb = joblib.load("technical_skills_mlb.pkl")

soft_model = joblib.load("soft_skills_model.pkl")
soft_mlb = joblib.load("soft_skills_mlb.pkl")

# ------------------------
# 🎛️ Dashboard Layout
# ------------------------
st.set_page_config(page_title="Job Insights Dashboard", layout="wide")
st.title("📊 Job Description Insights Dashboard")

# ------------------------
# 🚀 Input Mode Selection
# ------------------------
mode = st.radio("Choose input mode:", ["Single Job Description", "Multiple via CSV"])

if mode == "Single Job Description":
    desc = st.text_area("Paste the job description here:", height=250)

    if st.button("🔎 Analyze"):
        if not desc.strip():
            st.warning("Please enter a job description.")
        else:
            desc_input = [desc]

            majors_pred = majors_model.predict(desc_input)
            majors_output = majors_mlb.inverse_transform(majors_pred)[0]

            tech_pred = tech_model.predict(desc_input)
            tech_output = tech_mlb.inverse_transform(tech_pred)[0]

            soft_pred = soft_model.predict(desc_input)
            soft_output = soft_mlb.inverse_transform(soft_pred)[0]

            detected_languages = []
            language_keywords = ["python", "java", "c++", "c#", "sql", "r", "matlab", "javascript"]
            if "Programming (General)" in tech_output:
                text_lower = desc.lower()
                for lang in language_keywords:
                    if lang in text_lower:
                        detected_languages.append(lang.upper() if lang in ["r", "sql"] else lang.capitalize())

            st.markdown("### 📈 Prediction Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("🎓 Majors", len(majors_output))
            col2.metric("🛠️ Tech Skills", len(tech_output))
            col3.metric("💼 Soft Skills", len(soft_output))

            st.markdown("### 🧠 Predicted Categories")
            st.markdown("#### 🎓 Majors")
            st.info(", ".join(majors_output) if majors_output else "None")

            st.markdown("#### 🛠️ Technical Skills")
            st.success(", ".join(tech_output) if tech_output else "None")
            if "Programming (General)" in tech_output and detected_languages:
                st.markdown("🔍 **Detected Programming Languages:**")
                st.warning(", ".join(detected_languages))

            st.markdown("#### 💼 Soft Skills")
            st.success(", ".join(soft_output) if soft_output else "None")

            st.markdown("### 📊 Prediction Distribution")
            chart_data = pd.DataFrame({
                "Category": ["Majors", "Technical Skills", "Soft Skills"],
                "Count": [len(majors_output), len(tech_output), len(soft_output)]
            })
            st.bar_chart(chart_data.set_index("Category"))

            with st.expander("📅 Export Results"):
                export_df = pd.DataFrame({
                    "Majors": [", ".join(majors_output)],
                    "Technical Skills": [", ".join(tech_output)],
                    "Soft Skills": [", ".join(soft_output)],
                    "Languages Detected": [", ".join(detected_languages) if detected_languages else "None"]
                })
                st.dataframe(export_df)
                csv = export_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download as CSV", csv, "job_prediction.csv", "text/csv")

elif mode == "Multiple via CSV":
    uploaded_file = st.file_uploader("Upload a CSV file with a 'Description' column", type="csv")

    if uploaded_file:
        st.write("✅ File uploaded:", uploaded_file.name)  # DEBUGGING PRINT
        df = pd.read_csv(uploaded_file)
        if "Description" not in df.columns:
            st.error("CSV must contain a 'Description' column.")
        else:
            desc_list = df["Description"].fillna("").tolist()
            majors_pred = majors_model.predict(desc_list)
            majors_output = majors_mlb.inverse_transform(majors_pred)

            tech_pred = tech_model.predict(desc_list)
            tech_output = tech_mlb.inverse_transform(tech_pred)

            soft_pred = soft_model.predict(desc_list)
            soft_output = soft_mlb.inverse_transform(soft_pred)

            results = []
            for i, desc in enumerate(desc_list):
                detected_languages = []
                if "Programming (General)" in tech_output[i]:
                    text_lower = desc.lower()
                    for lang in ["python", "java", "c++", "c#", "sql", "r", "matlab", "javascript"]:
                        if lang in text_lower:
                            detected_languages.append(lang.upper() if lang in ["r", "sql"] else lang.capitalize())
                results.append({
                    "Description": desc[:100] + "..." if len(desc) > 100 else desc,
                    "Majors": ", ".join(majors_output[i]),
                    "Technical Skills": ", ".join(tech_output[i]),
                    "Soft Skills": ", ".join(soft_output[i]),
                    "Languages Detected": ", ".join(detected_languages) if detected_languages else "None"
                })

            result_df = pd.DataFrame(results)
            st.markdown("### 📋 Prediction Table")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download All Results as CSV", csv, "batch_predictions.csv", "text/csv")