# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Loan Risk Predictor", layout="centered")

# Title
st.title("💸 Loan Default & Repayment Predictor")
st.markdown("This mini project predicts whether a customer will default on a loan and estimates repayment duration using Machine Learning.")

# Sample Dataset
data = {
    'customer_id': [101, 102, 103, 104, 105, 106, 107, 108],
    'loan_amount': [10000, 20000, 15000, 30000, 12000, 5000, 25000, 18000],
    'interest_rate': [12.5, 15.2, 10.5, 18.0, 9.5, 8.0, 14.8, 13.2],
    'income': [40000, 50000, 35000, 80000, 30000, 25000, 60000, 45000],
    'credit_score': [700, 600, 720, 580, 710, 650, 590, 630],
    'previous_loans': [1, 2, 0, 3, 0, 0, 1, 2],
    'days_to_pay': [87, 94, 86, 106, 90, 87, 100, 96],
    'default': [0, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)
df['is_first_time'] = df['previous_loans'].apply(lambda x: 1 if x == 0 else 0)

features = ['loan_amount', 'interest_rate', 'income', 'credit_score', 'is_first_time']

# Scaling
scaler = StandardScaler()
X = df[features]
X_scaled = scaler.fit_transform(X)

# Train Classifier
clf = RandomForestClassifier()
clf.fit(X_scaled, df['default'])

# Train Regressor
reg = LinearRegression()
reg.fit(X_scaled, df['days_to_pay'])

# Sidebar Inputs
st.sidebar.header("📥 Enter Customer Details")

loan_amount = st.sidebar.slider("Loan Amount", 5000, 50000, 15000, step=500)
interest_rate = st.sidebar.slider("Interest Rate (%)", 5.0, 25.0, 12.0, step=0.1)
income = st.sidebar.slider("Annual Income", 20000, 100000, 45000, step=1000)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650, step=10)
is_first_time = st.sidebar.radio("Is First-Time Borrower?", ["Yes", "No"])

first_time_flag = 1 if is_first_time == "Yes" else 0

# Prepare input
input_data = pd.DataFrame([[loan_amount, interest_rate, income, credit_score, first_time_flag]],
                          columns=features)
input_scaled = scaler.transform(input_data)

# Predictions
default_pred = clf.predict(input_scaled)[0]
days_to_pay_pred = reg.predict(input_scaled)[0]

# Display Results
st.subheader("📊 Prediction Results")

if default_pred == 1:
    st.error("⚠️ This customer is **likely to default** on the loan.")
else:
    st.success("✅ This customer is **likely to repay** the loan.")

st.info(f"📅 Estimated Days to Repay: **{int(days_to_pay_pred)} days**")

# Feature Importance
st.subheader("🔍 Feature Importance")
importances = clf.feature_importances_
fig, ax = plt.subplots()
ax.barh(features, importances, color='skyblue')
ax.set_xlabel("Importance Score")
st.pyplot(fig)

# Show Raw Data
if st.checkbox("📄 Show Training Data"):
    st.write(df)

