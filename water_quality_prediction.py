

# import streamlit as st
# import re
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.impute import KNNImputer
# from whatsapp_alert import send_whatsapp_alert
# import os


# THRESHOLDS = {
#     "temperature": {"min": None, "max": 35},
#     "dissolved oxygen": {"min": 5, "max": None},
#     "ph": {"min": 6.5, "max": 8.5},
#     "conductivity": {"min": None, "max": 750},
#     "bod": {"min": None, "max": 3},
#     "nitrate": {"min": None, "max": 45},
#     "fecal coliform": {"min": None, "max": 500},
#     "total coliform": {"min": None, "max": 1000},
#     "fecal streptococci": {"min": None, "max": 100}
# }


# def runprediction_streamlit():

#     # ================= DATA LOADING =================
#     BASE_DIR = os.path.dirname(__file__)
#     path = os.path.join(BASE_DIR, "WQuality_River-Data-2023 (1).xlsx")

#     df_raw = pd.read_excel(path, header=None)

#     header_row = 0
#     for i in range(10):
#         row_texts = " ".join([str(v).lower() for v in df_raw.iloc[i].tolist() if pd.notna(v)])
#         if any(k in row_texts for k in ["temperature", "dissolved oxygen", "conductivity"]):
#             header_row = i
#             break

#     df = pd.read_excel(path, header=header_row)

#     # Clean column names
#     df.columns = df.columns.astype(str).str.strip().str.replace('\n', ' ', regex=True)

#     # Remove unnamed columns
#     df = df.loc[:, ~df.columns.str.contains("^Unnamed", case=False)]

#     # Drop fully empty rows
#     df = df.dropna(how="all")


#     # ================= COLUMN CLEANING =================
    
#     col_map = {}
#     for col in df.columns:
#         if "unnamed" in col.lower():
#             continue

#         n = re.sub(r'\(.*?\)', '', col.lower()).strip()
#         n = re.sub(r'[_\-\s]+', ' ', n)
#         col_map.setdefault(n, []).append(col)


#     rep_df = pd.DataFrame(index=df.index)
#     for base, cols in col_map.items():
#         numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors='coerce').notna().any()]
#         if numeric_cols:
#             rep_df[base] = pd.to_numeric(df[numeric_cols[0]], errors='coerce')

#     imputer = KNNImputer(n_neighbors=5)
#     df_clean = pd.DataFrame(imputer.fit_transform(rep_df), columns=rep_df.columns)


#     # ================= SAFETY CHECK =================
#     def check_safe_row(row):
#         for k, t in THRESHOLDS.items():
#             cols = [c for c in df_clean.columns if k in c.lower()]
#             if not cols:
#                 continue
#             val = row[cols[0]]
#             if (t["min"] is not None and val < t["min"]) or (t["max"] is not None and val > t["max"]):
#                 return 0
#         return 1

#     df_labels = df_clean.copy()
#     df_labels["Safe"] = df_labels.apply(check_safe_row, axis=1)

#     X = df_labels.drop(columns=["Safe"]).values
#     y = df_labels["Safe"].values

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # ================= MODEL =================
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42, stratify=y
#     )

#     X_train_g = np.expand_dims(X_train, -1)
#     n_nodes = X_train.shape[1]

#     X_in = Input(shape=(n_nodes, 1))
#     W = tf.Variable(tf.random.normal((n_nodes, n_nodes)), trainable=True)
#     x = Lambda(lambda x: tf.matmul(W, x))(X_in)
#     x = Dense(64, activation="relu")(x)
#     x = Dropout(0.4)(x)
#     x = Flatten()(x)
#     out = Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=X_in, outputs=out)
#     model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
#     model.fit(X_train_g, y_train, epochs=10, batch_size=8, verbose=0)

#     # ================= UI =================
    
#     st.header("🔍 Water Quality Prediction")

#     uservals = []
#     for col in df_clean.columns:
#         if "unnamed" in col.lower():
#             continue
#         uservals.append(st.number_input(col, value=float(df_clean[col].mean())))


#     # ================= CREATE SAMPLE =================
#     sample = np.array([uservals])
#     sample_series = pd.Series(uservals, index=df_clean.columns)

#     # ================= PREDICTION =================
#     rule_safe = check_safe_row(sample_series)
#     sample_scaled = scaler.transform(sample)
#     sample_scaled_g = np.expand_dims(sample_scaled, -1)
#     pred_prob = float(model.predict(sample_scaled_g)[0][0])
#     pred_wqi = 100 * np.mean(sample_scaled)


#     # ================= UNSAFE NOTES =================
#     unsafe_notes = []
#     for col in sample_series.index:
#         val = sample_series[col]
#         col_clean = col.lower().replace(" ", "").replace("_", "")
#         for key, t in THRESHOLDS.items():
#             if key in col_clean:
#                 if (t["min"] and val < t["min"]) or (t["max"] and val > t["max"]):
#                     reason = "below min" if (t["min"] and val < t["min"]) else "above max"
#                     unsafe_notes.append(f"{col}: {val} ({reason})")
 

#     # ================= ALERT LOGIC =================
    
#         # ================= ALWAYS SEND ALERT IF UNSAFE =================

#     if unsafe_notes:
#         alert_msg = (
#             "🚨 WATER QUALITY ALERT 🚨\n\n"
#             + "\n".join(unsafe_notes)
#          + "\n\nStatus: UNSAFE\nAction: Check water source immediately."
#      )

#         st.warning(alert_msg)
#         send_whatsapp_alert(alert_msg)


from xml.parsers.expat import model

# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
# from tensorflow.keras.optimizers import Adam
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# from serial_reader import read_sensor_data
# from email_alert import send_email_alert   # ✅ ADDED

# # ================= THRESHOLDS =================
# THRESHOLDS = {
#     "pH": {"min": 6.5, "max": 8.5},
#     "Turbidity": {"min": None, "max": 5},
#     "TDS": {"min": None, "max": 300},
#     "Temperature": {"min": None, "max": 35},
#     "DO": {"min": 5, "max": None},
#     "Ammonia": {"min": None, "max": 0.5},
#     "Conductivity": {"min": None, "max": 750}
# }

# # ================= RULE FUNCTION =================
# def check_safe(values):
#     for key, val in values.items():
#         t = THRESHOLDS[key]
#         if (t["min"] is not None and val < t["min"]) or \
#            (t["max"] is not None and val > t["max"]):
#             return 0
#     return 1

# # ================= TRAIN MODEL =================
# @st.cache_resource
# def train_model():

#     data = []
#     labels = []

#     for _ in range(1000):
#         sample = {
#             "pH": np.random.uniform(5, 9),
#             "Turbidity": np.random.uniform(0, 10),
#             "TDS": np.random.uniform(50, 600),
#             "Temperature": np.random.uniform(20, 40),
#             "DO": np.random.uniform(2, 10),
#             "Ammonia": np.random.uniform(0, 1),
#             "Conductivity": np.random.uniform(100, 1000)
#         }

#         label = check_safe(sample)
#         data.append(list(sample.values()))
#         labels.append(label)

#     X = np.array(data)
#     y = np.array(labels)

#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

#     # Neural Network
#     inp = Input(shape=(X.shape[1],))
#     x = Dense(64, activation="relu")(inp)
#     x = Dropout(0.3)(x)
#     x = Dense(32, activation="relu")(x)
#     out = Dense(1, activation="sigmoid")(x)

#     model = Model(inputs=inp, outputs=out)
#     model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
#     # Train model
#     history = model.fit(X_train, y_train, epochs=10, verbose=0)

# #  Get training accuracy
#     train_acc = history.history['accuracy'][-1]

# #  Evaluate on test data
#     loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

# # Print in terminal
#     print("Training Accuracy:", train_acc)
#     print("Test Accuracy:", test_acc)
#     return model, scaler


# # ================= STREAMLIT APP =================
# def runprediction_streamlit():

#     # Track email sent status
#     if "email_sent" not in st.session_state:
#         st.session_state.email_sent = False

#     st.header("💧 Water potability Prediction")

#     model, scaler = train_model()

#     # 🔥 SENSOR DATA
#     sensor = read_sensor_data()

#     if not sensor:
#         st.error("❌ No sensor data")
#         return

#     ph = sensor["pH"]
#     turbidity = sensor["Turbidity"]
#     temperature = sensor["Temperature"]
#     tds = sensor.get("TDS", 0)

#     st.subheader("📡 Sensor Data")
#     st.write(f"pH: {ph:.2f}")
#     st.write(f"Turbidity: {turbidity:.2f}")
#     st.write(f"Temperature: {temperature:.2f}")
#     st.write(f"TDS: {tds:.2f}")

#     # USER INPUTS
#     st.subheader("✍️ Additional Inputs")
#     do = st.number_input("DO", 0.0, 20.0, 8.0)
#     ammonia = st.number_input("Ammonia", 0.0, 1.0, 0.1)
#     conductivity = st.number_input("Conductivity", 0.0, 2000.0, 300.0)

#     if st.button("Predict Water Quality"):

#         sample = {
#             "pH": ph,
#             "Turbidity": turbidity,
#             "TDS": tds,
#             "Temperature": temperature,
#             "DO": do,
#             "Ammonia": ammonia,
#             "Conductivity": conductivity
#         }

#         # RULE CHECK
#         rule_result = check_safe(sample)

#         # ML PREDICTION
#         sample_arr = np.array([list(sample.values())])
#         sample_scaled = scaler.transform(sample_arr)
#         prob = model.predict(sample_scaled)[0][0]

#         # ================= ISSUES =================
#         unsafe_params_for_email = []

#         for key, val in sample.items():
#             t = THRESHOLDS[key]
#             if (t["min"] and val < t["min"]) or (t["max"] and val > t["max"]):
#                 unsafe_params_for_email.append({
#                     "parameter": key,
#                     "value": float(val)
#                 })

#         # RESULT
#         st.subheader("📊 Results")

#         if prob > 0.5:
#             st.success(f"✅ SAFE ")
#             st.session_state.email_sent = False
#         else:
#             st.error(f"🚨 NOT SAFE ")

#             # ✅ SEND EMAIL ONLY ONCE
#             if not st.session_state.email_sent:

#                 receivers = [
#                     "achsagracin@gmail.com",
#                     "afrahakim1234@gmail.com",
#                     "fathimas.nazar1234@gmail.com"
#                 ]

#                 send_email_alert(unsafe_params_for_email, "WQ-STATION-01", receivers)

#                 st.success("📩 ALERT EMAIL SENT")
#                 st.session_state.email_sent = True
#             else:
#                 st.info("ℹ️ Email already sent")

        
#         # REASONS
#         st.subheader("🔍 Issues Found")
#         for key, val in sample.items():
#             t = THRESHOLDS[key]
#             if (t["min"] and val < t["min"]) or (t["max"] and val > t["max"]):
#                 st.write(f"⚠ {key}: {val} out of range")

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from serial_reader import read_sensor_data
from email_alert import send_email_alert   # ✅ ADDED

# ================= THRESHOLDS =================
THRESHOLDS = {
    "pH": {"min": 6.5, "max": 8.5},
    "Turbidity": {"min": None, "max": 5},
    "TDS": {"min": None, "max": 300},
    "Temperature": {"min": None, "max": 35},
    "DO": {"min": 5, "max": None},
    "Ammonia": {"min": None, "max": 0.5},
    "Conductivity": {"min": None, "max": 750}
}

# ================= RULE FUNCTION =================
def check_safe(values):
    for key, val in values.items():
        t = THRESHOLDS[key]
        if (t["min"] is not None and val < t["min"]) or \
           (t["max"] is not None and val > t["max"]):
            return 0
    return 1

# ================= TRAIN MODEL =================
@st.cache_resource
def train_model():

    data = []
    labels = []

    for _ in range(1000):
        sample = {
            "pH": np.random.uniform(5, 9),
            "Turbidity": np.random.uniform(0, 10),
            "TDS": np.random.uniform(50, 600),
            "Temperature": np.random.uniform(20, 40),
            "DO": np.random.uniform(2, 10),
            "Ammonia": np.random.uniform(0, 1),
            "Conductivity": np.random.uniform(100, 1000)
        }

        label = check_safe(sample)
        data.append(list(sample.values()))
        labels.append(label)

    X = np.array(data)
    y = np.array(labels)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Neural Network
    inp = Input(shape=(X.shape[1],))
    x = Dense(64, activation="relu")(inp)
    x = Dropout(0.3)(x)
    x = Dense(32, activation="relu")(x)
    out = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])

    # Train model
    history = model.fit(X_train, y_train, epochs=10, verbose=0)

    # Accuracy calculations
    train_acc = history.history['accuracy'][-1]
    loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Terminal output (will show on first run)
    print("Training Accuracy:", train_acc)
    print("Test Accuracy:", test_acc)

    return model, scaler, train_acc, test_acc


# ================= STREAMLIT APP =================
def runprediction_streamlit():

    # Track email sent status
    if "email_sent" not in st.session_state:
        st.session_state.email_sent = False

    st.header("💧 Water potability Prediction")

    # Load (cached) model
    model, scaler, _, _ = train_model()

    # 🔥 SENSOR DATA
    sensor = read_sensor_data()

    if not sensor:
        st.error("❌ No sensor data")
        return

    ph = sensor["pH"]
    turbidity = sensor["Turbidity"]
    temperature = sensor["Temperature"]
    tds = sensor.get("TDS", 0)

    st.subheader("📡 Sensor Data")
    st.write(f"pH: {ph:.2f}")
    st.write(f"Turbidity: {turbidity:.2f}")
    st.write(f"Temperature: {temperature:.2f}")
    st.write(f"TDS: {tds:.2f}")

    # USER INPUTS
    st.subheader("✍️ Additional Inputs")
    do = st.number_input("DO", 0.0, 20.0, 8.0)
    ammonia = st.number_input("Ammonia", 0.0, 1.0, 0.1)
    conductivity = st.number_input("Conductivity", 0.0, 2000.0, 300.0)

    if st.button("Predict Water Quality"):

        sample = {
            "pH": ph,
            "Turbidity": turbidity,
            "TDS": tds,
            "Temperature": temperature,
            "DO": do,
            "Ammonia": ammonia,
            "Conductivity": conductivity
        }

        # RULE CHECK
        rule_result = check_safe(sample)

        # ML PREDICTION
        sample_arr = np.array([list(sample.values())])
        sample_scaled = scaler.transform(sample_arr)
        prob = model.predict(sample_scaled)[0][0]

        # ================= ISSUES =================
        unsafe_params_for_email = []

        for key, val in sample.items():
            t = THRESHOLDS[key]
            if (t["min"] and val < t["min"]) or (t["max"] and val > t["max"]):
                unsafe_params_for_email.append({
                    "parameter": key,
                    "value": float(val)
                })

        # RESULT
        st.subheader("📊 Results")

        if prob > 0.5:
            st.success(f"✅ SAFE ")
            st.session_state.email_sent = False
        else:
            st.error(f"🚨 NOT SAFE ")

            # ✅ SEND EMAIL ONLY ONCE
            if not st.session_state.email_sent:

                receivers = [
                    "achsagracin@gmail.com",
                    "afrahakim1234@gmail.com",
                    "fathimas.nazar1234@gmail.com"
                ]

                send_email_alert(unsafe_params_for_email, "WQ-STATION-01", receivers)

                st.success("📩 ALERT EMAIL SENT")
                st.session_state.email_sent = True
            else:
                st.info("ℹ️ Email already sent")

        # REASONS
        st.subheader("🔍 Issues Found")
        for key, val in sample.items():
            t = THRESHOLDS[key]
            if (t["min"] and val < t["min"]) or (t["max"] and val > t["max"]):
                st.write(f"⚠ {key}: {val} out of range")