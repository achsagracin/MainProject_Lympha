import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model
model = joblib.load("water_model.pkl")

# Page Config
st.set_page_config(
    page_title="Water Potability Dashboard",
    page_icon="💧",
    layout="wide"
)

# --- Global Dark Theme CSS ---
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0a192f !important;
        color: #e6f1ff !important;
    }
    [data-testid="stSidebar"] {
        background: #0d1b2a !important;
    }
    [data-testid="stSidebar"] * {
        color: #e6f1ff !important;
    }
    h1, h2, h3, h4, h5 {
        color: #64ffda !important;
    }
    label, .stMarkdown, .stNumberInput label {
        color: #64ffda !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border-radius: 10px;
        border: none;
        height: 3em;
        width: 100%;
        font-weight: 600;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #0072ff, #00c6ff);
        box-shadow: 0px 0px 10px #00c6ff;
    }
    .stSuccess {
        background-color: rgba(0, 128, 0, 0.25);
        border-left: 5px solid #00ff99;
        color: #e6f1ff !important;
    }
    .stError {
        background-color: rgba(255, 0, 0, 0.25);
        border-left: 5px solid #ff4d4d;
        color: #e6f1ff !important;
    }
    .stDataFrame, .stDataFrame table {
        background-color: #112240 !important;
        color: #e6f1ff !important;
    }
    .block-container {
        padding: 1rem 2rem;
    }
    </style>
""", unsafe_allow_html=True)


# --- Header ---
st.markdown("<h1 style='text-align: center;'>💧 Water Potability Dashboard</h1>", unsafe_allow_html=True)
st.write("<h4 style='text-align:center; color:#64ffda;'>AI-powered water safety prediction system</h4>", unsafe_allow_html=True)

# --- Sidebar ---
section = st.sidebar.radio("📊 Navigation", [
    "🏠 Home",
    "🔬 Test Sample",
    "📈 Data Insights",
    "⚖️ What-if Analysis",
    "🐟 Fish Predictor"
])


# --- Function: Get User Input ---
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

# --- HOME SECTION ---
if section == "🏠 Home":
    st.markdown("""
        <div style="text-align:center; color:#a8dadc;">
        <p>Welcome to the <b>Water Potability Dashboard</b>.<br>
        Use this dashboard to test your water samples or explore dataset insights.</p>
        </div>
    """, unsafe_allow_html=True)
    st.image(
        "https://assets-news.housing.com/news/wp-content/uploads/2018/12/24203624/Low-cost-pure-drinking-water-project-launched-around-West-Bengals-Shantiniketan-FB-1200x628-compressed.jpg",
        use_container_width=True
    )

# --- TEST SAMPLE SECTION ---
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

# --- DATA INSIGHTS SECTION ---
elif section == "📈 Data Insights":
    st.subheader("📈 Dataset Insights & Correlation Heatmap")
    try:
        df = pd.read_csv("water_potability.csv")
        st.write("Sample of the dataset:")
        st.dataframe(df.head(), width="stretch")

        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        plt.title("Feature Correlation Heatmap", color="white")
        ax.tick_params(colors="white")
        st.pyplot(fig, clear_figure=True)
    except:
        st.warning("Dataset not found. Please add `water_potability.csv` for insights.")

# --- WHAT-IF ANALYSIS SECTION ---
elif section == "⚖️ What-if Analysis":
    st.subheader("⚖️ What-if Simulation")
    st.write("Adjust one parameter and see prediction change:")
    user_input_df = get_user_input()

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
# --- FISH PREDICTOR SECTION ---
elif section == "🐟 Fish Predictor":
    st.subheader("🐟 Pond Water Fish Species Prediction")

    try:
        df_fish = pd.read_csv("pond_water_dataset_expanded.csv")

        # Train quick model (for demonstration)
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.ensemble import RandomForestClassifier

        X = df_fish[['ph', 'temperature', 'turbidity']]
        y = df_fish['fish']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_fish = RandomForestClassifier(n_estimators=100, random_state=42)
        model_fish.fit(X_train_scaled, y_train)

        # User Inputs
        st.write("Enter pond water parameters:")
        col1, col2, col3 = st.columns(3)
        with col1:
            ph = st.number_input("pH", 0.0, 14.0, 7.0)
        with col2:
            temp = st.number_input("Temperature (°C)", 0.0, 40.0, 25.0)
        with col3:
            turb = st.number_input("Turbidity (NTU)", 0.0, 100.0, 5.0)

        if st.button("🎣 Predict Fish Species"):
            sample = [[ph, temp, turb]]
            sample_scaled = scaler.transform(sample)
            predicted = model_fish.predict(sample_scaled)
            fish_name = label_encoder.inverse_transform(predicted)
            st.success(f"🐠 Predicted Fish Species: **{fish_name[0]}**")

    except Exception as e:
        st.error("Dataset not found or error occurred.")
        st.text(e)

