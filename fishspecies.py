import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report

@st.cache_resource(ttl=3600)
def load_and_train_model():
    try:
        df = pd.read_csv("Water_Quality_with_Fish.csv")  # Adjust path accordingly
    except FileNotFoundError:
        st.error("Dataset file 'Water_Quality_with_Fish.csv' not found. Please upload the file in your project folder.")
        st.stop()

    df['Survivable_Fish_List'] = df['Survivable_Fish'].apply(lambda x: [f.strip() for f in str(x).split(',')])
    X = df[['pH', 'Temperature (°C)', 'Turbidity (NTU)', 'Dissolved Oxygen (mg/L)', 'Conductivity (µS/cm)']]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(df['Survivable_Fish_List'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    trained_models = {}
    skipped_fish = []

    for i, fish in enumerate(mlb.classes_):
        y_fish = y_train[:, i]
        if len(np.unique(y_fish)) < 2:
            skipped_fish.append(fish)
            continue

        clf = VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=120, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.08, random_state=42))
            ],
            voting='soft'
        )
        clf.fit(X_train_scaled, y_fish)
        trained_models[fish] = clf

    return scaler, mlb, trained_models, skipped_fish

def fishspecies_streamlit():
    st.title("🐟 Fish Survival Prediction")

    scaler, mlb, trained_models, skipped_fish = load_and_train_model()

    st.write("Enter water quality parameters:")

    ph = st.number_input("pH value", min_value=0.0, max_value=14.0, value=7.0, format="%.2f")
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=40.0, value=20.0, format="%.2f")
    turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, max_value=100.0, value=5.0, format="%.2f")
    dissolved_oxygen = st.number_input("Dissolved Oxygen (mg/L)", min_value=0.0, max_value=20.0, value=8.0, format="%.2f")
    conductivity = st.number_input("Conductivity (µS/cm)", min_value=0.0, max_value=2000.0, value=150.0, format="%.2f")

    if st.button("Predict Survival"):
        user_input = pd.DataFrame([[ph, temperature, turbidity, dissolved_oxygen, conductivity]],
                                  columns=['pH', 'Temperature (°C)', 'Turbidity (NTU)',
                                           'Dissolved Oxygen (mg/L)', 'Conductivity (µS/cm)'])
        user_input_scaled = scaler.transform(user_input)

        fish_probs = []
        for fish, model in trained_models.items():
            if hasattr(model, 'predict_proba') and model.predict_proba(user_input_scaled).shape[1] > 1:
                prob = model.predict_proba(user_input_scaled)[0][1] * 100
            else:
                prob = 0.0
            fish_probs.append((fish, prob))

        for fish in skipped_fish:
            fish_probs.append((fish, 0.0))

        fish_probs.sort(key=lambda x: x[1], reverse=True)

        st.write("### Predicted Survival Probabilities:")
        for fish, prob in fish_probs:
            st.write(f"**{fish}**: {prob:.2f}% chance of survival")

        best_fish, best_prob = fish_probs[0]
        st.success(f"Most Likely to Survive: {best_fish} ({best_prob:.2f}%)")

        # Bar chart visualization
        fish_names = [fp[0] for fp in fish_probs]
        probabilities = [fp[1] for fp in fish_probs]
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=fish_names, y=probabilities, palette="crest", ax=ax)
        ax.set_xlabel("Fish Species")
        ax.set_ylabel("Survival Probability (%)")
        ax.set_title("Predicted Survival Probability per Fish Species")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
