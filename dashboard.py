# import streamlit as st
# import pandas as pd
# import joblib
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load model
# model = joblib.load("water_model.pkl")

# # Page config
# st.set_page_config(
#     page_title="Water Potability Dashboard",
#     page_icon="💧",
#     layout="wide"
# )

# # Custom CSS
# st.markdown("""
#     <style>
#     .main {
#         background-color: #f4f8fb;
#     }
#     .stButton button {
#         background-color: #4CAF50;
#         color: white;
#         border-radius: 10px;
#         height: 3em;
#         width: 100%;
#     }
#     .stSuccess {
#         background-color: #d7f9d9;
#     }
#     .stError {
#         background-color: #f9d7d7;
#     }
#     </style>
# """, unsafe_allow_html=True)

# # Dashboard Header
# st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💧 Water Potability Analysis Dashboard</h1>", unsafe_allow_html=True)
# st.write("### Ensure the safety of drinking water using AI-powered prediction")

# # Sidebar
# st.sidebar.header("📊 Navigation")
# section = st.sidebar.radio("Go to", ["🏠 Home", "🔬 Test Sample", "📈 Data Insights",])

# # Home Section
# if section == "🏠 Home":
#     st.image("https://assets-news.housing.com/news/wp-content/uploads/2018/12/24203624/Low-cost-pure-drinking-water-project-launched-around-West-Bengals-Shantiniketan-FB-1200x628-compressed.jpg", use_container_width=True)
#     st.markdown("""
#         Welcome to the **Water Potability Dashboard**.  
#         Use the sidebar to test your own sample or explore insights about the dataset.
#     """)

# # Test New Sample
# elif section == "🔬 Test Sample":
#     st.subheader("🔍 Enter Water Sample Parameters")

#     col1, col2, col3 = st.columns(3)
#     with col1:
#         ph = st.number_input("pH", 0.0, 14.0, 7.0)
#         hardness = st.number_input("Hardness", 0.0, 500.0, 150.0)
#         solids = st.number_input("Solids", 0.0, 50000.0, 20000.0)
#     with col2:
#         chloramines = st.number_input("Chloramines", 0.0, 20.0, 7.0)
#         sulfate = st.number_input("Sulfate", 0.0, 500.0, 350.0)
#         conductivity = st.number_input("Conductivity", 0.0, 1000.0, 400.0)
#     with col3:
#         organic_carbon = st.number_input("Organic Carbon", 0.0, 30.0, 15.0)
#         trihalo = st.number_input("Trihalomethanes", 0.0, 120.0, 80.0)
#         turbidity = st.number_input("Turbidity", 0.0, 10.0, 5.0)

#     if st.button("🚰 Predict Potability"):
#         sample = pd.DataFrame([{
#             "ph": ph, "Hardness": hardness, "Solids": solids,
#             "Chloramines": chloramines, "Sulfate": sulfate,
#             "Conductivity": conductivity, "Organic_carbon": organic_carbon,
#             "Trihalomethanes": trihalo, "Turbidity": turbidity
#         }])

#         prediction = model.predict(sample)[0]

#         placeholder = st.empty()
#         if prediction == 0:
#             with placeholder.container():
#                 st.error("⚠️ ALERT: Unsafe water detected – Do not consume")
#                 time.sleep(3)
#             placeholder.empty()
#         else:
#             with placeholder.container():
#                 st.success("✅ Safe water detected – Fit for consumption")
#                 time.sleep(3)
#             placeholder.empty()

# # Data Insights
# elif section == "📈 Data Insights":
#     st.subheader("📊 Dataset Insights & Correlation Heatmap")
    
#     # Load dataset (if available)
#     try:
#         df = pd.read_csv("water_potability.csv")
#         st.write("Sample of the dataset:")
#         st.dataframe(df.head())

#         # Heatmap
#         fig, ax = plt.subplots(figsize=(8,6))
#         sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
#         st.pyplot(fig)
#     except:
#         st.warning("Dataset not found. Please add `water_potability.csv` for insights.")


import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("water_model.pkl")

# Page config
st.set_page_config(
    page_title="Water Potability Dashboard",
    page_icon="💧",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {background-color: #f4f8fb;}
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stSuccess {background-color: #d7f9d9;}
    .stError {background-color: #f9d7d7;}
    </style>
""", unsafe_allow_html=True)

# Dashboard Header
st.markdown("<h1 style='text-align: center; color: #2E86C1;'>💧 Water Potability Dashboard</h1>", unsafe_allow_html=True)
st.write("### Ensure the safety of drinking water using AI-powered prediction")

# Sidebar Navigation
section = st.sidebar.radio("📊 Navigation", [
    "🏠 Home", 
    "🔬 Test Sample", 
    "📈 Data Insights", 
    "⚖️ What-if Analysis"
])

# Function to get user input
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

# --- Collect user input once ---
user_input_df = get_user_input()

# --- Home Section ---
if section == "🏠 Home":
    st.image(
        "https://assets-news.housing.com/news/wp-content/uploads/2018/12/24203624/Low-cost-pure-drinking-water-project-launched-around-West-Bengals-Shantiniketan-FB-1200x628-compressed.jpg",
        use_container_width=True
    )
    st.markdown("""
        Welcome to the **Water Potability Dashboard**.  
        Use the sidebar to test your own sample or explore dataset insights.
    """)

# --- Test Sample Section ---
elif section == "🔬 Test Sample":
    st.subheader("🔬 Test Sample Prediction")
    if st.button("🚰 Predict Potability"):
        prediction = model.predict(user_input_df)[0]
        prob = model.predict_proba(user_input_df)[0][1]

        if prediction == 1:
            st.success(f"✅ Safe water detected (Confidence: {prob*100:.1f}%)")
        else:
            st.error(f"⚠️ Unsafe water detected (Confidence: {(1-prob)*100:.1f}%)")

# --- Data Insights Section ---
elif section == "📈 Data Insights":
    st.subheader("📈 Dataset Insights & Correlation Heatmap")
    try:
        df = pd.read_csv("water_potability.csv")
        st.write("Sample of the dataset:")
        st.dataframe(df.head())

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)
    except:
        st.warning("Dataset not found. Please add `water_potability.csv` for insights.")

# --- What-if Analysis Section ---
elif section == "⚖️ What-if Analysis":
    st.subheader("⚖️ What-if Simulation")
    st.write("Adjust one parameter and see how the prediction changes:")

    feature = st.selectbox("Select feature to adjust", list(user_input_df.columns))
    new_value = st.slider(
        "New value for " + feature,
        float(user_input_df[feature].min()),
        float(user_input_df[feature].max())*2,
        float(user_input_df.iloc[0][feature])
    )

    modified_sample = user_input_df.copy()
    modified_sample[feature] = new_value

    col1, col2 = st.columns(2)

    with col1:
        pred = model.predict(user_input_df)[0]
        prob = model.predict_proba(user_input_df)[0][1]
        if pred == 1:
            st.success(f"✅ Original: Safe ({prob*100:.1f}%)")
        else:
            st.error(f"⚠️ Original: Unsafe ({(1-prob)*100:.1f}%)")

    with col2:
        pred_new = model.predict(modified_sample)[0]
        prob_new = model.predict_proba(modified_sample)[0][1]
        if pred_new == 1:
            st.success(f"✅ Modified: Safe ({prob_new*100:.1f}%)")
        else:
            st.error(f"⚠️ Modified: Unsafe ({(1-prob_new)*100:.1f}%)")
