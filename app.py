import streamlit as st
import pandas as pd
import pickle


# ================= LOAD MODEL =================
with open("machine_failure_model.pkl", "rb") as f:
    model = pickle.load(f)


st.title("🏭 Machine Failure Prediction App")
st.write("Enter machine sensor values:")


# ================= INPUTS =================

plant_location = st.selectbox("Plant Location", ["Plant A", "Plant B", "Plant C","Plant D"])


temperature = st.number_input("Temperature")
vibration = st.number_input("Vibration")
pressure = st.number_input("Pressure")
humidity = st.number_input("Humidity")
runtime_hours = st.number_input("Runtime Hours")
load_percentage = st.number_input("Load Percentage")
maintenance_history = st.number_input("Maintenance History")


# ================= CREATE DATAFRAME =================
input_data = pd.DataFrame([{
    "plant_location": plant_location,
    "temperature": temperature,
    "vibration": vibration,
    "pressure": pressure,
    "humidity": humidity,
    "runtime_hours": runtime_hours,
    "load_percentage": load_percentage,
    "maintenance_history": maintenance_history
}])


# ================= PREDICTION =================
if st.button("Predict Failure"):

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    st.subheader("Result")

    # probability of failure = class 1
    fail_index = list(model.classes_).index(1)
    risk_score = probability[fail_index] * 100

    if prediction == 1:
        st.error(f"⚠️ Machine Likely to FAIL\n\nRisk Score: {risk_score:.1f}%")
    else:
        st.success(f"✅ Machine Healthy\n\nRisk Score: {risk_score:.1f}%")

    st.progress(float(risk_score / 100))
