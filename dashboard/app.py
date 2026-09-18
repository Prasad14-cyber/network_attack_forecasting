import os
import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Network Attack Forecasting",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("🛡️ AI Network Attack Forecasting")

st.write(
    "AI-based network traffic analysis and attack forecasting dashboard."
)


# =========================================================
# LOAD MODEL AND SCALER
# =========================================================
# NOTE: app.py now lives inside dashboard/, while the model and scaler
# live inside models/ (a sibling folder). We build an absolute path
# based on this file's own location, so loading works correctly no
# matter where Streamlit is launched from (root folder, dashboard/
# folder, or elsewhere).

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "attack_forecasting_lstm.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "models", "scaler.pkl")


@st.cache_resource
def load_model_and_scaler():

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    scaler = joblib.load(
        SCALER_PATH
    )

    return model, scaler


try:
    model, scaler = load_model_and_scaler()

except Exception as e:
    st.error(
        f"Failed to load model or scaler: {e}\n\n"
        f"Expected model at: {MODEL_PATH}\n"
        f"Expected scaler at: {SCALER_PATH}"
    )
    st.stop()


st.success(
    "Model and scaler loaded successfully!"
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader("Model Information")

col1, col2 = st.columns(2)

with col1:

    st.write(
        "Model input shape:",
        model.input_shape
    )

with col2:

    st.write(
        "Number of features:",
        scaler.n_features_in_
    )


st.divider()


# =========================================================
# UPLOAD NETWORK TRAFFIC DATA
# =========================================================

st.header("📂 Upload Network Traffic Data")

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    # =====================================================
    # READ CSV
    # =====================================================

    df = pd.read_csv(uploaded_file)

    st.success(
        "CSV uploaded successfully!"
    )

    st.write(
        "Dataset shape:",
        df.shape
    )


    # =====================================================
    # CHECK FEATURES
    # =====================================================

    st.subheader(
        "🔍 Checking Dataset Features..."
    )

    feature_cols = [
        col
        for col in df.columns
        if col not in ["Timestamp", "Label"]
    ]

    st.write(
        "Features found:",
        len(feature_cols)
    )


    # Check feature count

    if len(feature_cols) != scaler.n_features_in_:

        st.error(
            f"Expected {scaler.n_features_in_} features, "
            f"but found {len(feature_cols)}."
        )

        st.stop()


    st.success(
        f"Feature count matches the trained model: {scaler.n_features_in_} features"
    )


    # =====================================================
    # SCALE NETWORK DATA
    # =====================================================

    X = df[feature_cols]

    try:

        X_scaled = scaler.transform(X)

    except Exception as e:

        st.error(
            f"Error while scaling data: {e}"
        )

        st.stop()


    st.success(
        "Network traffic data scaled successfully!"
    )

    st.write(
        "Scaled data shape:",
        X_scaled.shape
    )


    # =====================================================
    # CREATE TIME SEQUENCES
    # =====================================================

    SEQ_LEN = 10

    X_sequences = []

    for i in range(
        len(X_scaled) - SEQ_LEN
    ):

        sequence = X_scaled[
            i:i + SEQ_LEN
        ]

        X_sequences.append(
            sequence
        )


    X_sequences = np.array(
        X_sequences
    ).astype(
        np.float32
    )


    st.success(
        "10-step sequences created successfully!"
    )

    st.write(
        "Sequence shape:",
        X_sequences.shape
    )


    # =====================================================
    # CHECK WHETHER ENOUGH DATA EXISTS
    # =====================================================

    if len(X_sequences) == 0:

        st.error(
            "Not enough rows to create a 10-step sequence."
        )

        st.stop()


    # =====================================================
    # ATTACK PREDICTION
    # =====================================================

    st.subheader(
        "🚨 Attack Prediction"
    )

    with st.spinner(
        "AI model is analyzing network traffic..."
    ):

        y_prob = model.predict(
            X_sequences,
            verbose=0
        ).flatten()


    # Threshold = 0.5

    y_pred = (
        y_prob >= 0.5
    ).astype(
        int
    )


    st.success(
        "Attack predictions generated successfully!"
    )

    st.write(
        "Number of predictions:",
        len(y_pred)
    )


    # =====================================================
    # PREDICTION SUMMARY
    # =====================================================

    st.subheader(
        "📊 Prediction Summary"
    )


    total_predictions = len(
        y_pred
    )


    predicted_attacks = int(
        np.sum(
            y_pred == 1
        )
    )


    predicted_benign = int(
        np.sum(
            y_pred == 0
        )
    )


    average_probability = float(
        np.mean(
            y_prob
        )
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Total Predictions",
            f"{total_predictions:,}"
        )


    with col2:

        st.metric(
            "Predicted Attacks",
            f"{predicted_attacks:,}"
        )


    with col3:

        st.metric(
            "Predicted Benign",
            f"{predicted_benign:,}"
        )


    with col4:

        st.metric(
            "Average Attack Probability",
            f"{average_probability:.2%}"
        )


    # =====================================================
    # PROBABILITY TIMELINE
    # =====================================================

    st.subheader(
        "📈 Network Attack Probability Timeline"
    )


    if "Timestamp" in df.columns:

        timestamps = pd.to_datetime(
            df["Timestamp"]
            .iloc[SEQ_LEN:]
            .reset_index(drop=True)
        )


        timeline_df = pd.DataFrame({

            "Timestamp": timestamps,

            "Attack Probability": y_prob

        })


        timeline_df = timeline_df.set_index(
            "Timestamp"
        )


        st.line_chart(
            timeline_df[
                "Attack Probability"
            ]
        )


    else:

        st.warning(
            "Timestamp column not found. "
            "Showing predictions by sequence number."
        )


        timeline_df = pd.DataFrame({

            "Attack Probability": y_prob

        })


        st.line_chart(
            timeline_df[
                "Attack Probability"
            ]
        )


    # =====================================================
    # LATEST PREDICTION
    # =====================================================

    st.subheader(
        "🔔 Latest Prediction"
    )


    latest_probability = float(
        y_prob[-1]
    )


    if latest_probability >= 0.5:

        st.error(

            f"⚠️ Possible Attack Detected — "
            f"Attack Probability: "
            f"{latest_probability:.2%}"

        )

    else:

        st.success(

            f"✅ Traffic Appears Benign — "
            f"Attack Probability: "
            f"{latest_probability:.2%}"

        )


    # =====================================================
    # PREDICTION RESULTS TABLE
    # =====================================================

    st.subheader(
        "📋 Prediction Results"
    )


    if "Timestamp" in df.columns:

        result_timestamps = pd.to_datetime(

            df["Timestamp"]
            .iloc[SEQ_LEN:]
            .reset_index(drop=True)

        )

    else:

        result_timestamps = range(
            len(y_prob)
        )


    result_df = pd.DataFrame({

        "Timestamp":
            result_timestamps,

        "Attack Probability":
            y_prob,

        "Prediction":
            np.where(
                y_pred == 1,
                "Attack",
                "Benign"
            )

    })


    st.dataframe(
        result_df.head(20),
        use_container_width=True
    )


    # =====================================================
    # DOWNLOAD RESULTS
    # =====================================================

    st.subheader(
        "💾 Download Prediction Results"
    )


    csv_data = result_df.to_csv(
        index=False
    )


    st.download_button(

        label="Download Predictions CSV",

        data=csv_data,

        file_name="network_attack_predictions.csv",

        mime="text/csv"

    )