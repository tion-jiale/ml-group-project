import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ===============================
# Load Model and Scaler
# ===============================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ===============================
# App Config
# ===============================
st.set_page_config(
    page_title="Influential Disease Prediction",
    page_icon="🦠",
    layout="centered"
)

# ===============================
# Title & Description
# ===============================
st.title("🦠 Influential Disease Prediction by State")
st.markdown("""
This application predicts the **most influential disease in a state**  
based on health indicators using a trained **Machine Learning model**.
""")

# ===============================
# Sidebar
# ===============================
st.sidebar.header("📍 Select State")
state = st.sidebar.selectbox(
    "Choose a State",
    [
        "Alabama", "Alaska", "Arizona", "Arkansas", "California",
        "Colorado", "Connecticut", "Delaware", "Florida", "Georgia",
        "Hawaii", "Illinois", "Indiana", "Iowa", "Kansas"
        # add all states as needed
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Provide health indicators to predict the dominant disease.")

# ===============================
# Input Features
# ===============================
st.subheader("📊 Health Indicators")

# ⚠️ Replace feature names with your actual dataset features
feature_1 = st.number_input("Disease Incidence Rate", min_value=0.0, step=0.1)
feature_2 = st.number_input("Mortality Rate", min_value=0.0, step=0.1)
feature_3 = st.number_input("Hospitalization Rate", min_value=0.0, step=0.1)
feature_4 = st.number_input("Population Affected (%)", min_value=0.0, step=0.1)

# Combine features
input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

# ===============================
# Prediction
# ===============================
if st.button("🔍 Predict Influential Disease"):
    try:
        # Scale input
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled).max()

        # ⚠️ Replace with your actual disease labels
        disease_labels = {
            0: "Diabetes",
            1: "Heart Disease",
            2: "Cancer",
            3: "Respiratory Disease"
        }

        predicted_disease = disease_labels.get(int(prediction[0]), "Unknown")

        # ===============================
        # Output
        # ===============================
        st.success(f"### 🧬 Predicted Influential Disease in {state}")
        st.markdown(f"""
        **Disease:** `{predicted_disease}`  
        **Prediction Confidence:** `{probability:.2%}`
        """)

    except Exception as e:
        st.error("⚠️ Prediction failed. Please check inputs.")
        st.exception(e)

# ===============================
# Footer
# ===============================
st.markdown("---")
st.caption("Developed using Machine Learning & Streamlit")
