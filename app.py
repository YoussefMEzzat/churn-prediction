import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Churn Predictor", page_icon="📉")


@st.cache_resource
def load_model():
    return joblib.load("models/churn_model.joblib")


model = load_model()

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to estimate how likely they are to leave.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

with col2:
    st.subheader("Services")
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple = "No phone service" if phone == "No" else st.selectbox("Multiple Lines", ["No", "Yes"])

    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    addons = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
              "TechSupport", "StreamingTV", "StreamingMovies"]
    addon_values = {}
    for name in addons:
        if internet == "No":
            addon_values[name] = "No internet service"
        else:
            addon_values[name] = st.selectbox(name, ["No", "Yes"])

customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    **addon_values,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,
}])

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    proba = model.predict_proba(customer)[0, 1]

    st.metric("Churn probability", f"{proba:.0%}")

    if proba >= 0.7:
        st.error("🔴 High risk: reach out with a retention offer.")
    elif proba >= 0.4:
        st.warning("🟡 Medium risk: monitor this customer.")
    else:
        st.success("🟢 Low risk: customer is likely to stay.")