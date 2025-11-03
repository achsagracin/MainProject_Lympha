import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import prediction function
from water_quality_prediction import run_prediction

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
    "🏠 Dashboard": '<i class="fa-solid fa-house"></i> Dashboard',
    "🔍 Water Quality Prediction": '<i class="fa-solid fa-water"></i> Water Quality Prediction',
    "📈 Data Insights": '<i class="fa-solid fa-chart-line"></i> Data Insights'
}

section = st.sidebar.radio(
    "Navigation",
    list(menu.keys()),
    label_visibility="collapsed",
    key="menu_select"
)

# ==============================
# HEADER
# ==============================
st.markdown("""
<div class="topbar">
    <span class="title-left">💧 Water Potability Dashboard</span>
    <span class="title-right">AI-powered water safety prediction system</span>
</div>
""", unsafe_allow_html=True)

# ==============================
# SECTION HANDLING
# ==============================
if section == "🏠 Dashboard":
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

elif section == "🔍 Water Quality Prediction":
    from water_quality_prediction import run_prediction_streamlit
    run_prediction_streamlit()
