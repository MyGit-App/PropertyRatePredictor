
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("GradientBoostingTuned.pkl")
scaler = joblib.load("feature_scaler.pkl")

st.title("House Price Prediction (in Lacs)")

posted_by = st.selectbox("Posted By", ["Owner", "Dealer", "Builder"])
bhk_no = st.number_input("BHK Number", 1, 10, 2)
bhk_or_rk = st.selectbox("BHK or RK", ["BHK", "RK"])
sqft = st.number_input("Square Feet", 200, 10000, 1200)

under_construction = st.selectbox("Under Construction", [0, 1])
rera = st.selectbox("RERA Approved", [0, 1])
ready_to_move = st.selectbox("Ready To Move", [0, 1])
resale = st.selectbox("Resale", [0, 1])

longitude = st.number_input("Longitude", format="%.6f")
latitude = st.number_input("Latitude", format="%.6f")

if st.button("Predict Price"):

    # Same encoding used in notebook
    posted_by_map = {
        "Owner": 0,
        "Dealer": 1,
        "Builder": 2
    }

    bhk_or_rk_map = {
        "BHK": 0,
        "RK": 1
    }

    input_df = pd.DataFrame({
        "POSTED_BY": [posted_by_map[posted_by]],
        "UNDER_CONSTRUCTION": [under_construction],
        "RERA": [rera],
        "BHK_NO.": [bhk_no],
        "BHK_OR_RK": [bhk_or_rk_map[bhk_or_rk]],
        "SQUARE_FT": [sqft],
        "READY_TO_MOVE": [ready_to_move],
        "RESALE": [resale],
        "LONGITUDE": [longitude],
        "LATITUDE": [latitude]
    })

    # Apply same scaling as training
    input_scaled = scaler.transform(input_df)

    # Prediction
    prediction = model.predict(input_scaled)[0]

    st.success(f"Predicted Price: ₹ {prediction:.2f} Lacs")
