import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

df = pd.read_csv("diabetes.csv")
zero_cols = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

st.title("Diabetes Prediction App")
st.markdown("Enter patient details below to predict diabetes outcome.")

col1, col2 = st.columns(2)
with col1:
    pregnancies    = st.number_input("Pregnancies", 0, 20, 1)
    glucose        = st.number_input("Glucose", 0, 300, 120)
    blood_pressure = st.number_input("Blood Pressure", 0, 200, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi     = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age     = st.number_input("Age", 1, 120, 30)

if st.button("Predict"):
    input_data   = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    input_scaled = scaler.transform(input_data)
    prediction   = model.predict(input_scaled)[0]
    probability  = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"Result: Diabetic   |   Probability: {probability:.2%}")
    else:
        st.success(f"Result: Not Diabetic   |   Probability of Diabetes: {probability:.2%}")