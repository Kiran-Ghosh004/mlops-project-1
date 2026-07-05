"""
Streamlit Frontend for Telecom Churn Prediction
--------------------------------------------------
A simple form-based UI that collects customer data and sends it to the
FastAPI service for a churn prediction.
"""
import requests
import streamlit as st
import os

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📉", layout="centered")

st.title("📉 Telecom Customer Churn Predictor")
st.markdown(
    "Enter a customer's details below to predict their likelihood of churning. "
    "This uses a tuned XGBoost model served via FastAPI."
)

with st.form("customer_form"):
    st.subheader("Customer Info")
    col1, col2 = st.columns(2)

    with col1:
        customerID = st.text_input("Customer ID", value="0000-DEMO")
        gender = st.selectbox("Gender", ["Female", "Male"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Has Partner", ["Yes", "No"])
        Dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])

    with col2:
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=29.85, step=0.5)
        TotalCharges = st.number_input("Total Charges", min_value=0.0, value=29.85, step=0.5)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    payload = {
        "customerID": customerID,
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction Result")

        risk = result["risk_label"]
        prob = result["churn_probability"]

        if risk == "High Risk":
            st.error(f"⚠️ {risk} — Churn Probability: {prob:.1%}")
        elif risk == "Medium Risk":
            st.warning(f"⚡ {risk} — Churn Probability: {prob:.1%}")
        else:
            st.success(f"✅ {risk} — Churn Probability: {prob:.1%}")

        st.progress(min(prob, 1.0))
        st.json(result)

    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure the FastAPI server is running on port 8000.")
    except requests.exceptions.HTTPError as e:
        st.error(f"API returned an error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")