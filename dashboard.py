
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing prediction modules
from water_quality_prediction import runprediction_streamlit
from fishspecies import fishspecies_streamlit
st.markdown("""
    <style>
        section[data-testid="stSidebar"] > div:first-child {
            width: 300px !important;
        }
        /* Prevent text wrapping and keep everything on one line */
        [data-testid="stSidebar"] div[role="radiogroup"] label p {
            white-space: nowrap !important;
            overflow: hidden;
            text-overflow: ellipsis;
            font-size: 15.5px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============================== PAGE CONFIG ==============================
st.set_page_config(page_title="Water Potability Dashboard", page_icon="💧", layout="wide")

# ============================== LOAD CSS & Font ==============================
with open("ui_style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">',
    unsafe_allow_html=True
)

# ============================== Sidebar Menu ==============================
menu = {
    "🏠 Dashboard": "Dashboard",
    "🔍 Water Quality Prediction": "Water Quality Prediction",
    "🐟 Fish Survival Prediction": "Fish Survival Prediction"
}
section = st.sidebar.radio("Navigation", list(menu.keys()), label_visibility="collapsed", key="menu_select")

# ============================== Header ==============================
st.markdown("""
<div class="topbar">
    <span class="title-left">Water Potability Dashboard</span>
    <span class="title-right">AI-powered water safety prediction system</span>
</div>
""", unsafe_allow_html=True)

# ============================== Dashboard Widgets ==============================
if section == "🏠 Dashboard":
    st.markdown("""
<div class="card-container">
    <div class="metric-card">
        <div class="metric-label">Total Samples Tested</div>
        <div class="metric-value">1250</div>
        <div class="metric-sub">Since January 2025</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Safe Water Samples</div>
        <div class="metric-value">880</div>
        <div class="metric-sub">70.4% Potability Rate</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Unsafe Water Samples</div>
        <div class="metric-value">370</div>
        <div class="metric-sub">Detected as Non-potable</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Model Accuracy</div>
        <div class="metric-value">94%</div>
        <div class="metric-sub">Based on Test Dataset</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Last Updated</div>
        <div class="metric-value">03 Nov 2025</div>
        <div class="metric-sub">08:45 AM</div>
    </div>
</div>
""", unsafe_allow_html=True)

elif section == "🔍 Water Quality Prediction":
    runprediction_streamlit()

elif section == "🐟 Fish Survival Prediction":
    fishspecies_streamlit()
