import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ---------------------
# Load Model & Scaler
# ---------------------
model = joblib.load("best_wind_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------
# Streamlit UI
# ---------------------
st.title("🌬️ Wind Speed Prediction App")
st.write("Enter weather details and get the predicted wind speed.")

# ---------------------
# User Input Fields (Original Names)
# ---------------------
rain = st.number_input("🌧️ RAIN (mm)", value=0.0)
tmax = st.number_input("🌡️ T.MAX (Max Temperature °C)", value=25.0)
tmin = st.number_input("❄️ T.MIN (Min Temperature °C)", value=18.0)
tmin_g = st.number_input("📉 T.MIN.G (Ground Min Temp °C)", value=17.0)
date = st.date_input("📅 Select Date", datetime.today())

# ---------------------
# Convert Date Features
# ---------------------
year = date.year
month = date.month
day = date.day

# Convert input to dataframe (same structure as training)
input_data = pd.DataFrame([{
    "RAIN": rain,
    "T.MAX": tmax,
    "T.MIN": tmin,
    "T.MIN.G": tmin_g,
    "Year": year,
    "Month": month,
    "Day": day
}])

# ---------------------
# Prediction Button
# ---------------------
if st.button("🔍 Predict Wind Speed"):
    
    # Scale data
    scaled_data = scaler.transform(input_data)

    # Predict
    prediction = model.predict(scaled_data)[0]

    st.success(f"💨 Predicted Wind Speed: **{round(prediction, 2)} km/h**")
