import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==============================
# LOAD MODEL
# ==============================
model = joblib.load("water_model.pkl")

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Water Potability Dashboard", page_icon="💧", layout="wide")

# ==============================
# LOAD CUSTOM CSS
# ==============================
with open("ui_style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ==============================
# LOAD FONT AWESOME
# ==============================
st.markdown(
    '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">',
    unsafe_allow_html=True
)

# ==============================
# SIDEBAR MENU
# ==============================
menu = {
    "🏠 Home": '<i class="fa-solid fa-house"></i> Home',
    "🔬 Test Sample": '<i class="fa-solid fa-vial"></i> Test Sample',
    "📈 Data Insights": '<i class="fa-solid fa-chart-line"></i> Data Insights'
}

section = st.sidebar.radio(
    "Navigation",
    list(menu.keys()),
    label_visibility="collapsed",
    key="menu_select"
)

# Sidebar Styling
st.sidebar.markdown("""
<style>
[data-testid="stSidebar"] {
    background-color: #0f0f0f !important;
    border-right: 1px solid #222;
    padding-top: 1rem;
}
[data-testid="stSidebar"] .stRadio > div {
    display: flex;
    flex-direction: column;
    gap: 0.8rem;
}
[data-testid="stSidebar"] label {
    background-color: #1a1a1a !important;
    color: #dcdcdc !important;
    padding: 10px 16px;
    border-radius: 8px;
    font-size: 15px;
    font-weight: 500;
    transition: all 0.25s ease;
}
[data-testid="stSidebar"] label:hover {
    background-color: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER / TOPBAR
# ==============================
st.markdown("""
<div class="topbar">
    <span class="title-left">💧 Water Potability Dashboard</span>
    <span class="title-right">AI-powered water safety prediction system</span>
</div>
""", unsafe_allow_html=True)

# ==============================
# FUNCTION: USER INPUT
# ==============================
def get_user_input():
    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.number_input("pH", 0.0, 14.0, 7.0)
        hardness = st.number_input("Hardness", 0.0, 500.0, 150.0)
        solids = st.number_input("Solids", 0.0, 50000.0, 20000.0)
    with col2:
        chloramines = st.number_input("Chloramines", 0.0, 20.0, 7.0)
        sulfate = st.number_input("Sulfate", 0.0, 500.0, 350.0)
        conductivity = st.number_input("Conductivity", 0.0, 1000.0, 400.0)
    with col3:
        organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 15.0)
        trihalo = st.number_input("Trihalomethanes", 0.0, 120.0, 80.0)
        turbidity = st.number_input("Turbidity", 0.0, 10.0, 5.0)
    return pd.DataFrame([{
        "ph": ph, "Hardness": hardness, "Solids": solids,
        "Chloramines": chloramines, "Sulfate": sulfate,
        "Conductivity": conductivity, "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalo, "Turbidity": turbidity
    }])

# ==============================
# HOME SECTION
# ==============================
# ===============================
# HOME SECTION
# ===============================
if section == "🏠 Home":
    # --- Metric Cards ---
    st.markdown("""
        <div class="card-container">
            <div class="metric-card">
                <div class="metric-label">Test samples this month</div>
                <div class="metric-value">450</div>
                <div class="metric-sub">Automated predictions</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Safe Water Rate</div>
                <div class="metric-value">71%</div>
                <div class="metric-sub">Across last 6 months</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Model Accuracy</div>
                <div class="metric-value">94%</div>
                <div class="metric-sub">AI Confidence level</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # --- Description ---
    st.markdown(
        "<div class='home-text'>Welcome to the <b>Water Potability Dashboard</b>. Use this dashboard to test samples, analyze data trends, and explore potability insights.</div>",
        unsafe_allow_html=True
    )

    # --- Analytics-style Summary Box ---
    st.markdown("""
    <div class="summary-panel">
        <div class="summary-title">Water Quality Overview</div>
        <div class="summary-subtitle">Summary for the last 3 months</div>
        <div class="summary-stats">
            <div class="stat">
                <div class="stat-label">Average pH</div>
                <div class="stat-value">7.2</div>
            </div>
            <div class="stat">
                <div class="stat-label">Average Turbidity</div>
                <div class="stat-value">4.6 NTU</div>
            </div>
            <div class="stat">
                <div class="stat-label">Average Conductivity</div>
                <div class="stat-value">385 µS/cm</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==============================
# TEST SAMPLE SECTION
# ==============================
elif section == "🔬 Test Sample":
    st.subheader("🔬 Test Sample Prediction")
    user_input_df = get_user_input()
    if st.button("🚰 Predict Potability"):
        prediction = model.predict(user_input_df)[0]
        prob = model.predict_proba(user_input_df)[0][1]
        if prediction == 1:
            st.success(f"✅ Safe water detected (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"⚠️ Unsafe water detected (Confidence: {(1-prob)*100:.1f}%)")

# ==============================
# DATA INSIGHTS SECTION
# ==============================
elif section == "📈 Data Insights":
    st.subheader("📈 Dataset Insights & Correlation Heatmap")
    try:
        df = pd.read_csv("water_potability.csv")
        st.dataframe(df.head())
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        plt.title("Feature Correlation Heatmap", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig)
    except:
        st.warning("Dataset not found. Please add `water_potability.csv` for insights.")
