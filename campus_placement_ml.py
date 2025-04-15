import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load trained pipeline model
model = joblib.load("gradient_boosting_model.joblib")

# Set Streamlit page layout
st.set_page_config(page_title="Campus Placement Prediction", layout="wide")
st.markdown(
    "<h1 style='color: green;'>🎓 Campus Placement Prediction App</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "Predict whether a student will get placed based on academic records and background."
)


# Load dataset (optional, only for EDA)
@st.cache_data
def load_data():
    return pd.read_csv("Placement.csv")  # <-- Replace with actual dataset path


df = load_data()

# Collect Inputs
st.subheader("📝 Enter Student Details")
col1, col2, col3, col4 = st.columns(4)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    ssc_p = st.number_input("SSC %", 0.0, 100.0, step=0.1)
    ssc_b = st.selectbox("SSC Board", ["Central", "Others"])
    salary = st.number_input(
        "Enter Salary (₹)", 0, 2000000, step=1000
    )  # User input for salary


with col2:
    hsc_p = st.number_input("HSC %", 0.0, 100.0, step=0.1)
    hsc_b = st.selectbox("HSC Board", ["Central", "Others"])
    hsc_s = st.selectbox("HSC Stream", ["Commerce", "Science", "Arts"])

with col3:
    degree_p = st.number_input("Degree %", 0.0, 100.0, step=0.1)
    degree_t = st.selectbox("Degree Type", ["Sci&Tech", "Comm&Mgmt", "Others"])
    workex = st.selectbox("Work Experience", ["Yes", "No"])

with col4:
    etest_p = st.number_input("E-test %", 0.0, 100.0, step=0.1)
    specialisation = st.selectbox("Specialisation", ["Mkt&HR", "Mkt&Fin"])
    mba_p = st.number_input("MBA %", 0.0, 100.0, step=0.1)

# Prediction button
if st.button("🚀 Predict Placement"):
    input_df = pd.DataFrame(
        [
            {
                "gender": 1 if gender == "Male" else 0,
                "ssc_p": ssc_p,
                "ssc_b": ssc_b,
                "hsc_p": hsc_p,
                "hsc_b": hsc_b,
                "hsc_s": hsc_s,
                "degree_p": degree_p,
                "degree_t": degree_t,
                "workex": workex,
                "etest_p": etest_p,
                "specialisation": specialisation,
                "mba_p": mba_p,
                "salary": salary,  # User input salary
            }
        ]
    )

    # Predict
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

    if prediction == "Placed":
        st.success(
            f"🎉 The student is **Likely to be Placed** (Confidence: {max(proba) * 100:.2f}%)"
        )
        st.markdown(f"📊 **Entered Salary**: ₹{salary:,.2f}")
    else:
        st.error(
            f"❌ The student is **Not Likely to be Placed** (Confidence: {max(proba) * 100:.2f}%)"
        )
        st.markdown(f"📊 **Entered Salary**: ₹{salary:,.2f}")

# --- Expandable EDA Section ---
with st.expander("📊 Click to Explore Placement Trends (EDA)"):
    st.subheader("📈 Placement vs Degree Percentage")
    fig1, ax1 = plt.subplots()
    sns.boxplot(x="status", y="degree_p", data=df, palette="Set2", ax=ax1)
    st.pyplot(fig1)

    st.subheader("📊 Work Experience vs Placement")
    fig2, ax2 = plt.subplots()
    sns.countplot(x="workex", hue="status", data=df, palette="pastel", ax=ax2)
    st.pyplot(fig2)

    st.subheader("📊 Specialisation Distribution")
    fig3, ax3 = plt.subplots()
    sns.countplot(x="specialisation", hue="status", data=df, palette="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.subheader("📊 Correlation Heatmap")
    corr = df.select_dtypes(include="number").corr()
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="Greens", ax=ax4)
    st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Tariku")
