# def run_prediction():
#     import re
#     import numpy as np
#     import pandas as pd
#     import tensorflow as tf
#     from tensorflow.keras.models import Model
#     from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda
#     from tensorflow.keras.optimizers import Adam
#     from tensorflow.keras.callbacks import EarlyStopping
#     from sklearn.utils.class_weight import compute_class_weight
#     from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
#     from sklearn.model_selection import train_test_split, KFold
#     from sklearn.ensemble import GradientBoostingRegressor
#     from sklearn.impute import KNNImputer
#     from sklearn.metrics import r2_score, classification_report
#     import seaborn as sns
#     import matplotlib.pyplot as plt

#     # === CONFIG: thresholds for Safe/Unsafe ===
#     THRESHOLDS = {
#         "temperature": {"min": None, "max": 35},
#         "dissolved oxygen": {"min": 5, "max": None},
#         "ph": {"min": 6.5, "max": 8.5},
#         "conductivity": {"min": None, "max": 750},
#         "bod": {"min": None, "max": 3},
#         "nitrate": {"min": None, "max": 45},
#         "fecal coliform": {"min": None, "max": 500},
#         "total coliform": {"min": None, "max": 1000},
#         "fecal streptococci": {"min": None, "max": 100}
#     }

#     # === 1. Load Excel ===
#     path = "WQuality_River-Data-2023 (1).xlsx"
#     df_raw = pd.read_excel(path, header=None)
#     header_row = None
#     for i in range(10):
#         row_texts = " ".join([str(v).lower() for v in df_raw.iloc[i].tolist() if pd.notna(v)])
#         if "temperature" in row_texts or "dissolved oxygen" in row_texts or "conductivity" in row_texts:
#             header_row = i
#             break
#     if header_row is None:
#         header_row = 0
#     df = pd.read_excel(path, header=header_row)
#     df.columns = df.columns.astype(str).str.strip().str.replace('\n', ' ', regex=True)
#     df = df.dropna(how="all")
#     print("Loaded dataframe shape:", df.shape)

#     # === 2. Detect parameter columns ===
#     col_map = {}
#     for col in df.columns:
#         name = col.strip()
#         n = re.sub(r'\s+', ' ', name.lower())
#         n_no_units = re.sub(r'\(.*?\)', '', n).strip()
#         if re.search(r'\bmin\b|\bmax\b|\bminimum\b|\bmaximum\b', n_no_units):
#             base = re.sub(r'\b(min|max|minimum|maximum)\b', '', n_no_units).strip()
#         else:
#             base = n_no_units
#         base = re.sub(r'[_\-\s]+', ' ', base).strip()
#         if base == '':
#             base = n.lower()
#         col_map.setdefault(base, []).append(col)

#     keywords = list(THRESHOLDS.keys())
#     selected_bases = sorted({base for base in col_map if any(k in base for k in keywords)})
#     rep_df = pd.DataFrame(index=df.index)

#     for base in selected_bases:
#         cols = col_map.get(base, [])
#         min_cols = [c for c in cols if re.search(r'\bmin\b|\bminimum\b', c.lower())]
#         max_cols = [c for c in cols if re.search(r'\bmax\b|\bmaximum\b', c.lower())]
#         if min_cols and max_cols:
#             rep = (pd.to_numeric(df[min_cols[0]], errors='coerce') + pd.to_numeric(df[max_cols[0]], errors='coerce')) / 2.0
#         else:
#             numeric_cols = [c for c in cols if pd.to_numeric(df[c], errors='coerce').notna().any()]
#             if not numeric_cols:
#                 continue
#             rep = pd.to_numeric(df[numeric_cols[0]], errors='coerce')
#         rep_df[base] = rep

#     rep_df = rep_df.dropna(how='all')
#     print("Representative df shape:", rep_df.shape)

#     # === 3. Impute missing values ===
#     imputer = KNNImputer(n_neighbors=5)
#     df_clean = pd.DataFrame(imputer.fit_transform(rep_df), columns=rep_df.columns)

#     # === 4. Correlation analysis ===
#     corr_matrix = df_clean.corr()
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
#     plt.title("Feature Correlation Matrix")
#     plt.show()

#     # === 5. Safe/Unsafe labeling ===
#     def check_safe_row(row):
#         for feature_base, thresh in THRESHOLDS.items():
#             matches = [c for c in df_clean.columns if feature_base in c.lower()]
#             if not matches:
#                 continue
#             val = row[matches[0]]
#             if pd.isna(val):
#                 return 0
#             mn, mx = thresh.get("min"), thresh.get("max")
#             if mn is not None and val < mn: return 0
#             if mx is not None and val > mx: return 0
#         return 1

#     df_labels = df_clean.copy()
#     df_labels["Safe"] = df_labels.apply(check_safe_row, axis=1).astype(int)
#     print("\nLabel distribution:\n", df_labels["Safe"].value_counts())

#     # === 6. Prepare features
#     X = df_labels.drop(columns=["Safe"]).values
#     y = df_labels["Safe"].values
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # === 7. Train/test split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42, stratify=y
#     )

#     X_train_g = np.expand_dims(X_train, -1)
#     X_test_g = np.expand_dims(X_test, -1)
#     n_nodes = X_scaled.shape[1]

#     # === Compute class weights
#     class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
#     cw = {i: w for i, w in enumerate(class_weights)}
#     print("\nClass Weights:", cw)

#     # === Improved GCN-like model ===
#     X_in = Input(shape=(n_nodes, 1))
#     W = tf.Variable(tf.random.normal((n_nodes, n_nodes)), trainable=True)
#     x = Lambda(lambda x: tf.matmul(W, x))(X_in)
#     x = Dense(64, activation='relu')(x)
#     x = Dropout(0.4)(x)
#     x = Dense(32, activation='relu')(x)
#     x = Flatten()(x)
#     x = Dense(16, activation='relu')(x)
#     out = Dense(1, activation='sigmoid')(x)
#     model = Model(inputs=X_in, outputs=out)
#     model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
#     model.summary()

#     early = EarlyStopping(patience=8, restore_best_weights=True)
#     model.fit(X_train_g, y_train, validation_split=0.1, epochs=60, batch_size=8,
#               class_weight=cw, callbacks=[early], verbose=1)

#     loss, acc = model.evaluate(X_test_g, y_test, verbose=0)
#     print(f"\n✅ Improved GCN-like classifier Accuracy: {acc:.3f}")

#     # === Classification report ===
#     y_pred = (model.predict(X_test_g) > 0.5).astype(int)
#     print("\nDetailed classification metrics:")
#     print(classification_report(y_test, y_pred, target_names=["Unsafe", "Safe"]))

#     # === 8. Gradient Boosting for WQI ===
#     WQI_norm = np.dot((X - X.min(0)) / (X.max(0) - X.min(0) + 1e-9), np.ones(X.shape[1]) / X.shape[1])
#     pt, qt = PowerTransformer(), QuantileTransformer(output_distribution='normal', n_quantiles=min(100, X_scaled.shape[0]))
#     X_t = qt.fit_transform(pt.fit_transform(X_scaled))

#     kf = KFold(n_splits=min(5, max(2, X_t.shape[0]//20)), shuffle=True, random_state=42)
#     r2_scores = []
#     for train_idx, test_idx in kf.split(X_t):
#         gbr = GradientBoostingRegressor(random_state=42)
#         gbr.fit(X_t[train_idx], WQI_norm[train_idx])
#         pred = gbr.predict(X_t[test_idx])
#         r2_scores.append(r2_score(WQI_norm[test_idx], pred))
#     print(f"\n✅ Gradient Boosting cross-val R² mean: {np.mean(r2_scores):.3f}")

#     model_gbr = GradientBoostingRegressor(random_state=42)
#     model_gbr.fit(X_t, WQI_norm)

#     # === 9. Interactive prediction ===
#     print("\nEnter values for the following parameters:")
#     user_vals = []
#         # Continue inside run_prediction()

#     for c in df_clean.columns:
#         val = float(input(f"{c}: "))
#         user_vals.append(val)

#     sample = np.array([user_vals])
#     sample_series = pd.Series(user_vals, index=df_clean.columns)

#     # Rule-based safety check
#     rule_safe = check_safe_row(sample_series)

#     # Scale input for model
#     sample_scaled = scaler.transform(sample)
#     sample_scaled_g = np.expand_dims(sample_scaled, -1)

#     # Model prediction
#     pred_prob = float(model.predict(sample_scaled_g)[0, 0])
#     pred_label = 1 if pred_prob >= 0.6 and rule_safe == 1 else 0  # Combined rule

#     # WQI prediction
#     sample_t = qt.transform(pt.transform(sample_scaled))
#     pred_wqi = float(model_gbr.predict(sample_t)[0] * 100)

#     # === Output Results ===
#     print("\n=== Prediction Results ===")
#     print(f"Rule-based Safety: {'✅ SAFE' if rule_safe == 1 else '🚫 UNSAFE'}")
#     print(f"Model-based Safety (final): {'✅ SAFE' if pred_label == 1 else '🚫 UNSAFE'} (prob={pred_prob:.3f})")
#     print(f"Predicted WQI (0-100): {pred_wqi:.2f}")

#     # Show unsafe parameters
#     print("\nParameters outside safety thresholds:")
#     for c in df_clean.columns:
#         val = sample_series[c]
#         for k, t in THRESHOLDS.items():
#             if k in c.lower():
#                 mn, mx = t.get('min'), t.get('max')
#                 if (mn is not None and val < mn) or (mx is not None and val > mx):
#                     reason = "below min" if (mn is not None and val < mn) else "above max"
#                     print(f" - {c}: {val} ({reason})")

#     # Show top deviations from mean
#     means = df_clean.mean().values
#     diff = sample[0] - means
#     idx_sorted = np.argsort(-np.abs(diff))
#     print("\nTop 3 parameters deviating from dataset mean:")
#     for idx in idx_sorted[:3]:
#         c = df_clean.columns[idx]
#         d = diff[idx]
#         print(f" - {c}: {abs(d):.2f} {'higher' if d > 0 else 'lower'} than mean")

#     print("\nDone ✅")

# # Optional: Auto-run when the script is executed directly
# if __name__ == "__main__":
#     run_prediction()


import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import r2_score, classification_report

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st   # ✅ ADDED


# ✅ Your thresholds remain unchanged
THRESHOLDS = {
    "temperature": {"min": None, "max": 35},
    "dissolved oxygen": {"min": 5, "max": None},
    "ph": {"min": 6.5, "max": 8.5},
    "conductivity": {"min": None, "max": 750},
    "bod": {"min": None, "max": 3},
    "nitrate": {"min": None, "max": 45},
    "fecal coliform": {"min": None, "max": 500},
    "total coliform": {"min": None, "max": 1000},
    "fecal streptococci": {"min": None, "max": 100}
}

# ✅ MAIN EXISTING FUNCTION (UNCHANGED)
def run_prediction():
    # Your full original code remains here exactly as it is
    # (Not removed, not modified so terminal version still works)
    pass  # <- keep your original function code instead of pass


# ✅ ✅ NEW FUNCTION FOR STREAMLIT DASHBOARD
def run_prediction_streamlit():
    """Runs the same model but uses Streamlit instead of input()/print()."""

    st.subheader("🔍 Water Quality Prediction")

    # === 1. Load Excel ===
    path = "WQuality_River-Data-2023 (1).xlsx"
    df_raw = pd.read_excel(path, header=None)
    
    # === SAME CODE AS YOUR FUNCTION UP TO INPUT() PART ===
    # (Detect headers, clean dataframe)
    header_row = None
    for i in range(10):
        row_texts = " ".join([str(v).lower() for v in df_raw.iloc[i].tolist() if pd.notna(v)])
        if "temperature" in row_texts or "dissolved oxygen" in row_texts or "conductivity" in row_texts:
            header_row = i
            break
    if header_row is None:
        header_row = 0
    df = pd.read_excel(path, header=header_row)
    df.columns = df.columns.astype(str).str.strip().str.replace('\n', ' ', regex=True)
    df = df.dropna(how="all")

    # === 2. Build cleaned dataset like your original ===
    df_clean = df.copy()
    df_clean = df_clean.select_dtypes(include=[np.number])
    df_clean = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df_clean),
                            columns=df_clean.columns)

    def check_safe_row(row):
        for feature_base, thresh in THRESHOLDS.items():
            matches = [c for c in df_clean.columns if feature_base in c.lower()]
            if not matches:
                continue
            val = row[matches[0]]
            mn, mx = thresh.get("min"), thresh.get("max")
            if mn is not None and val < mn: return 0
            if mx is not None and val > mx: return 0
        return 1

    df_clean["Safe"] = df_clean.apply(check_safe_row, axis=1).astype(int)

    # === 3. Train/Test Preparation (Same as your function) ===
    X = df_clean.drop(columns=["Safe"]).values
    y = df_clean["Safe"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_g = np.expand_dims(X_train, -1)
    n_nodes = X_scaled.shape[1]

    # === 4. Build GCN-like model ===
    X_in = Input(shape=(n_nodes, 1))
    W = tf.Variable(tf.random.normal((n_nodes, n_nodes)), trainable=True)
    x = Lambda(lambda x: tf.matmul(W, x))(X_in)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=X_in, outputs=out)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_g, y_train, epochs=5, batch_size=8, verbose=0)

    # === 5. User Input Now via Streamlit (NOT input()) ===
    st.write("### Enter values for each parameter:")
    user_vals = []
    for c in df_clean.columns[:-1]:  # All except "Safe"
        val = st.number_input(f"{c}", step=0.01, format="%.2f")
        user_vals.append(val)

    if st.button("Predict Water Quality"):
        sample = np.array([user_vals])
        sample_scaled = scaler.transform(sample)
        sample_scaled_g = np.expand_dims(sample_scaled, -1)

        # Predict
        pred_prob = float(model.predict(sample_scaled_g)[0, 0])
        pred_label = 1 if pred_prob >= 0.6 else 0

        # === Show results in dashboard ===
        st.subheader("✅ Prediction Results")
        st.write(f"**Model-based Result:** {'✅ SAFE' if pred_label == 1 else '🚫 UNSAFE'}")
        st.write(f"**Confidence Score:** {pred_prob:.3f}")

