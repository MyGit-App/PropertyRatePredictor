
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("GradientBoostingTuned.pkl")
scaler = joblib.load("feature_scaler.pkl")
def show():
    st.title("🏠 Single Property Prediction")


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏠 Property Details")

        posted_by = st.selectbox("Posted By",
                                ["Owner","Dealer","Builder"])

        bhk_no = st.number_input("BHK Number",1,10,2)

        bhk_or_rk = st.selectbox("BHK or RK",
                                ["BHK","RK"])

        sqft = st.number_input("Square Feet",
                            200,10000,1200)

    with col2:
        st.subheader("📌 Property Status")

        under_construction = st.selectbox(
            "Under Construction",[0,1])

        rera = st.selectbox(
            "RERA",[0,1])

        ready_to_move = st.selectbox(
            "Ready To Move",[0,1])

        resale = st.selectbox(
            "Resale",[0,1])

    st.subheader("🌍 Location")

    col3, col4 = st.columns(2)

    with col3:
        longitude = st.number_input(
            "Longitude",
            format="%.6f"
        )

    with col4:
        latitude = st.number_input(
            "Latitude",
            format="%.6f"
        )


    if st.button("Predict Price"):

        posted_by_map = {
            "Owner":0,
            "Dealer":1,
            "Builder":2
        }

        bhk_map = {
            "BHK":0,
            "RK":1
        }

        inp = pd.DataFrame({
            "POSTED_BY":[posted_by_map[posted_by]],
            "UNDER_CONSTRUCTION":[under_construction],
            "RERA":[rera],
            "BHK_NO.":[bhk_no],
            "BHK_OR_RK":[bhk_map[bhk_or_rk]],
            "SQUARE_FT":[sqft],
            "READY_TO_MOVE":[ready_to_move],
            "RESALE":[resale],
            "LONGITUDE":[longitude],
            "LATITUDE":[latitude]
        })

        inp_scaled = scaler.transform(inp)

        prediction = model.predict(inp_scaled)[0]

        lower = prediction * 0.92
        upper = prediction * 1.08

        st.success(
            f"Predicted Price : ₹ {prediction:,.2f} Lacs"
        )

        st.info(
            f"Confidence Range : ₹ {lower:,.2f} - ₹ {upper:,.2f}"
        )

        st.subheader("Property Summary")

        st.write(f"BHK : {bhk_no}")

        st.write(f"Area : {sqft} Sq Ft")

        st.write(f"Ready To Move : {ready_to_move}")
