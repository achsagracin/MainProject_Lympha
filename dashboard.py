import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Page config (must be first)
st.set_page_config(page_title="LYMPHA", page_icon="💧", layout="wide")

# 2) Pages
from water_quality_prediction import runprediction_streamlit
from fishspecies import fishspecies_streamlit
from forecasting.ts_forecasting_ui import ts_forecasting_streamlit

# 3) Theme CSS
with open("ui_style.css", "r", encoding="utf-8") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.markdown(
    '<link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css" rel="stylesheet">',
    unsafe_allow_html=True
)

# 4) Sidebar (clean—no search, no logout, no toggle, no subtext)
st.sidebar.markdown("""
<div class="sb-card sb-brand">
  <div class="sb-avatar">LY</div>
  <div class="sb-brand-text">
    <div class="sb-brand-title">LYMPHA</div>
  </div>
</div>
""", unsafe_allow_html=True)

menu = {
    "🏠 Dashboard": "Dashboard",
    "🔍 Water Potability": "Water Potability",
    "🐟 Aquatic Habitability": "Aquatic Habitability",
    "📈 Time-Series Forecasting": "Time-Series Forecasting",
}
section = st.sidebar.radio(
    "Navigation", list(menu.keys()),
    label_visibility="collapsed", key="menu_select"
)

# 5) Brand bar (ONE bar only)
st.markdown("""
<div class="brandbar">
  <div class="brand">
    <svg viewBox="0 0 24 24" fill="#4f8df5" xmlns="http://www.w3.org/2000/svg" width="22" height="22">
      <path d="M12 2C12 2 6 8.5 6 12.5C6 16.09 8.91 19 12.5 19C16.09 19 19 16.09 19 12.5C19 8.5 12 2 12 2Z"/>
    </svg>
    <span>LYMPHA</span>
  </div>
  <div class="tagline">AI-powered water insights & forecasting</div>
</div>
""", unsafe_allow_html=True)

def page_title(text: str):
    st.markdown(f"<div class='page-title'>{text}</div>", unsafe_allow_html=True)

# 6) Pages
if section == "🏠 Dashboard":
    page_title("🏠 Dashboard")
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
          <div class="metric-sub">On test dataset</div>
      </div>
      <div class="metric-card">
          <div class="metric-label">Last Updated</div>
          <div class="metric-value">03 Nov 2025</div>
          <div class="metric-sub">08:45 AM</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1], gap="large")
    with col1:
        st.markdown("<div class='panel'><h4>Recent Measurements</h4><p style='color:#9aa4b2'>Connect your source to render a table or charts here.</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='panel'><h4>Alerts</h4><p style='color:#9aa4b2'>Add threshold-based alerts here.</p></div>", unsafe_allow_html=True)

elif section == "🔍 Water Potability":
    runprediction_streamlit()

elif section == "🐟 Aquatic Habitability":
     fishspecies_streamlit()

elif section == "📈 Time-Series Forecasting":
    ts_forecasting_streamlit()
