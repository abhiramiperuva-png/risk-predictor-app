import streamlit as st
import pandas as pd
import joblib

# Load your trained model (exported from Colab with joblib.dump)
model = joblib.load("risk_model.pkl")

# Define features
NUMERIC_FEATURES = ['annual_income','debt_to_income','loan_amount','interest_rate',
                    'term','delinq_2y','months_since_last_delinq','num_historical_failed_to_pay',
                    'public_record_bankrupt','tax_liens']
CATEGORICAL_FEATURES = ['homeownership','verified_income','loan_purpose','application_type','region']
REGION_MAP = {'NY':'North','NJ':'North','CA':'West','TX':'South','FL':'South','IL':'East','PA':'East'}

# Title
st.title("Customer Risk Category Predictor")

# Sidebar inputs
annual_income = st.sidebar.number_input("Annual Income", min_value=0, value=50000)
debt_to_income = st.sidebar.number_input("Debt-to-Income Ratio", min_value=0.0, value=20.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, value=15000)
interest_rate = st.sidebar.number_input("Interest Rate (%)", min_value=0.0, value=12.0)
term = st.sidebar.selectbox("Loan Term (months)", [36, 60])
homeownership = st.sidebar.selectbox("Homeownership", ["OWN","RENT","MORTGAGE"])
verified_income = st.sidebar.selectbox("Verified Income", ["Verified","Not Verified","Source Verified"])
loan_purpose = st.sidebar.selectbox("Loan Purpose", ["debt_consolidation","medical","small_business","other"])
application_type = st.sidebar.selectbox("Application Type", ["individual","joint"])
state = st.sidebar.text_input("State (e.g., NY)", "NY")

# Predict button
if st.sidebar.button("Predict Risk Category"):
    new_customer = {
        'annual_income': annual_income,
        'debt_to_income': debt_to_income,
        'loan_amount': loan_amount,
        'interest_rate': interest_rate,
        'term': term,
        'delinq_2y': 0,
        'months_since_last_delinq': -1,
        'num_historical_failed_to_pay': 0,
        'public_record_bankrupt': 0,
        'tax_liens': 0,
        'homeownership': homeownership,
        'verified_income': verified_income,
        'loan_purpose': loan_purpose,
        'application_type': application_type,
        'state': state
    }

    new_df = pd.DataFrame([new_customer])
    new_df['region'] = new_df['state'].map(REGION_MAP)
    new_df = new_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    probs = model.predict_proba(new_df)[0]
    classes = model.classes_
    prediction = model.predict(new_df)[0]

    st.subheader("Predicted Category")
    st.write(prediction)

    st.subheader("Class Probabilities")
    prob_df = pd.DataFrame({"Category": classes, "Probability": probs})
    st.bar_chart(prob_df.set_index("Category"))
