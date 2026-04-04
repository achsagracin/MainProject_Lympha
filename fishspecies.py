

# import streamlit as st
# import pandas as pd
# import numpy as np
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
# from sklearn.multioutput import MultiOutputClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score, f1_score, hamming_loss

# st.set_page_config(page_title="🐟 Aquatic Survivability Prediction", layout="centered")

# # =========================
# # LOAD & TRAIN MODEL
# # =========================
# @st.cache_resource
# def load_and_train():
#     BASE_DIR = os.path.dirname(__file__)
#     DATA_PATH = os.path.join(BASE_DIR, "synthetic_gan_full_dataset.csv")

#     if not os.path.exists(DATA_PATH):
#         st.error("Dataset file 'synthetic_gan_full_dataset.csv' not found.")
#         st.stop()

#     df = pd.read_csv(DATA_PATH)

#     df['Survivable_Fish_List'] = df['Fish'].apply(
#         lambda x: [f.strip() for f in str(x).split(',')]
#     )

#     X = df[['Temperature', 'pH', 'Turbidity', 'DO', 'Ammonia']]
#     mlb = MultiLabelBinarizer()
#     y = mlb.fit_transform(df['Survivable_Fish_List'])

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)

#     base_model = GradientBoostingClassifier(
#         n_estimators=300,
#         learning_rate=0.05,
#         max_depth=4,
#         random_state=42
#     )

#     model = MultiOutputClassifier(base_model)
#     model.fit(X_train_scaled, y_train)

#     # Evaluation
#     y_pred = model.predict(X_test_scaled)
#     metrics = {
#         "subset_acc": accuracy_score(y_test, y_pred),
#         "hamming_acc": 1 - hamming_loss(y_test, y_pred),
#         "micro_f1": f1_score(y_test, y_pred, average='micro'),
#         "macro_f1": f1_score(y_test, y_pred, average='macro')
#     }

#     return model, scaler, mlb, metrics


# # =========================
# # STREAMLIT APP FUNCTION
# # =========================
# def fishspecies_streamlit():
#     st.title("🐟 Aquatic Habitability Prediction")

#     model, scaler, mlb, metrics = load_and_train()

#     st.markdown("### Enter Water Quality Parameters")

#     temperature = st.number_input("Temperature (°C)", 0.0, 40.0, 20.0)
#     ph = st.number_input("pH", 0.0, 14.0, 7.0)
#     turbidity = st.number_input("Turbidity (NTU)", 0.0, 100.0, 5.0)
#     do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 8.0)
#     ammonia = st.number_input("Ammonia (mg/L)", 0.0, 1.0, 0.1)

#     if st.button("Predict Fish Survivability"):
#         user_df = pd.DataFrame(
#             [[temperature, ph, turbidity, do, ammonia]],
#             columns=['Temperature', 'pH', 'Turbidity', 'DO', 'Ammonia']
#         )

#         user_scaled = scaler.transform(user_df)

#         fish_probs = []
#         for i, fish in enumerate(mlb.classes_):
#             prob = model.estimators_[i].predict_proba(user_scaled)[0][1] * 100
#             fish_probs.append((fish, prob))

#         fish_probs.sort(key=lambda x: x[1], reverse=True)

#         st.subheader("🐠 Predicted Survival Probabilities")
#         for fish, prob in fish_probs:
#             st.write(f"**{fish}** : {prob:.2f}%")

#         best_fish, best_prob = fish_probs[0]
#         st.success(f"Most Likely to Survive: **{best_fish} ({best_prob:.2f}%)**")

#     st.markdown("---")
#     st.subheader("📊 Model Performance (Test Data)")
#     st.write(f"**Subset Accuracy (Exact Match):** {metrics['subset_acc'] * 100:.2f}%")
#     st.write(f"**Hamming Accuracy (Overall):** {metrics['hamming_acc'] * 100:.2f}%")
#     st.write(f"**Micro F1 Score:** {metrics['micro_f1'] * 100:.2f}%")
#     st.write(f"**Macro F1 Score:** {metrics['macro_f1'] * 100:.2f}%")


# # =========================
# # DIRECT RUN SUPPORT
# # =========================
# if __name__ == "__main__":
#     fishspecies_streamlit()


import streamlit as st
import pandas as pd
import numpy as np
import os
from serial_reader import read_sensor_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier

# =========================
# LOAD & TRAIN MODEL
# =========================
@st.cache_resource
def load_and_train():
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "synthetic_gan_full_dataset.csv")

    df = pd.read_csv(DATA_PATH)

    df['Survivable_Fish_List'] = df['Fish'].apply(
        lambda x: [f.strip() for f in str(x).split(',')]
    )

    X = df[['Temperature', 'pH', 'Turbidity', 'DO', 'Ammonia']]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['Survivable_Fish_List'])

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = MultiOutputClassifier(
        GradientBoostingClassifier(n_estimators=200, random_state=42)
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler, mlb


# =========================
# STREAMLIT FUNCTION
# =========================
def fishspecies_streamlit():
    st.title("🐟 Aquatic Habitability (Hybrid Input)")

    model, scaler, mlb = load_and_train()

    # 🔥 SENSOR DATA
    sensor_data = read_sensor_data()

    if not sensor_data:
        st.error("❌ No sensor data received")
        return


    temperature = sensor_data["Temperature"]
    ph = sensor_data["pH"]
    turbidity = sensor_data["Turbidity"]

    # 📊 Display sensor values
    st.subheader("📡 Sensor Values")
    st.write(f"🌡 Temperature: {temperature:.2f}")
    st.write(f"🧪 pH: {ph:.2f}")
    st.write(f"💧 Turbidity: {turbidity:.2f}")

    # ✍️ USER INPUTS
    st.subheader("✍️ Enter Remaining Parameters")

    do = st.number_input("Dissolved Oxygen (mg/L)", 0.0, 20.0, 8.0)
    ammonia = st.number_input("Ammonia (mg/L)", 0.0, 1.0, 0.1)

    # 🔥 PREDICTION
    if st.button("Predict Fish Survival"):

        user_df = pd.DataFrame(
            [[temperature, ph, turbidity, do, ammonia]],
            columns=['Temperature', 'pH', 'Turbidity', 'DO', 'Ammonia']
        )

        user_scaled = scaler.transform(user_df)

        fish_probs = []
        for i, fish in enumerate(mlb.classes_):
            prob = model.estimators_[i].predict_proba(user_scaled)[0][1] * 100
            fish_probs.append((fish, prob))

        fish_probs.sort(key=lambda x: x[1], reverse=True)

        st.subheader("🐠 Predicted Fish Survival")

        for fish, prob in fish_probs:
            st.write(f"**{fish}** : {prob:.2f}%")

        best_fish, best_prob = fish_probs[0]
        st.success(f"🏆 Best Fish: {best_fish} ({best_prob:.2f}%)")
