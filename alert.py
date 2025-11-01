# alert_system.py
import joblib
import pandas as pd

# Load trained model
model = joblib.load("water_model.pkl")

# Example new sample (can change values to test)
sample = pd.DataFrame([{
    "ph": 9.2,
    "Hardness": 150,
    "Solids": 20000,
    "Chloramines": 7.5,
    "Sulfate": 350,
    "Conductivity": 400,
    "Organic_carbon": 15,
    "Trihalomethanes": 80,
    "Turbidity": 5
}])

prediction = model.predict(sample)[0]

if prediction == 0:
    print("⚠️ ALERT: Unsafe water detected – Do not consume")
else:
    print("✅ Safe water detected – Fit for consumption")
