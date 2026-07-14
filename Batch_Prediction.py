
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.express as px

model = joblib.load("GradientBoostingTuned.pkl")
scaler = joblib.load("feature_scaler.pkl")
def show():
    st.title("📂 Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload CSV File",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df)
        
        df['POSTED_BY'] = df['POSTED_BY'].map({'Owner': 0, 'Dealer': 1, 'Builder': 2})

        df['BHK_OR_RK'] = df['BHK_OR_RK'].map({'BHK': 0, 'RK': 1})


        if 'ADDRESS' in df.columns:
            df.drop(['ADDRESS'], axis=1, inplace=True)
        
        
        # Separate target if available
        y_test = None

        if "TARGET(PRICE_IN_LACS)" in df.columns:
            y_test = df["TARGET(PRICE_IN_LACS)"]

        X_test = df.drop(
            columns=["TARGET(PRICE_IN_LACS)"],
            errors="ignore"
            )
        
        st.write(X_test.columns.tolist())
        # Ensure same feature order as training
        feature_order = [
                'POSTED_BY',
                'UNDER_CONSTRUCTION',
                'RERA',
                'BHK_NO.',
                'BHK_OR_RK',
                'SQUARE_FT',
                'READY_TO_MOVE',
                'RESALE',
                'LONGITUDE',
                'LATITUDE'
            ]

        X_test = X_test[feature_order]

        # Scale only features
        X_scaled = scaler.transform(X_test)

        try:

            predictions = model.predict(X_scaled)

            df["Predicted_Price_Lacs"] = predictions

            st.success("Prediction Complete")

            st.dataframe(df.head())

            st.download_button(
                "Download Predictions",
                df.to_csv(index=False),
                "Predictions.csv"
            )

            st.subheader("Prediction Distribution")

            fig, ax = plt.subplots(figsize=(8,4))

            ax.hist(
                predictions,
                bins=20,
                color="royalblue"
            )

            ax.set_xlabel("Price (Lacs)")
            ax.set_ylabel("Count")

            st.pyplot(fig)

            st.subheader("Prediction Metrics")

            col1,col2,col3 = st.columns(3)

            col1.metric(
                "Minimum",
                f"{predictions.min():.2f}"
            )

            col2.metric(
                "Average",
                f"{predictions.mean():.2f}"
            )

            col3.metric(
                "Maximum",
                f"{predictions.max():.2f}"
            )
        
            fig = px.histogram(
                df,
                x="Predicted_Price_Lacs",
                nbins=30,
                title="Prediction Distribution"
            )

            st.plotly_chart(fig)

            
            fig = px.box(
                df,
                y="Predicted_Price_Lacs",
                color_discrete_sequence=["orange"]
            )

            st.plotly_chart(fig)


            df["Segment"] = pd.cut(
                df["Predicted_Price_Lacs"],
                bins=[0,50,100,200,500,1000],
                labels=[
                    "Budget",
                    "Mid",
                    "Premium",
                    "Luxury",
                    "Ultra Luxury"
                ]
            )

            fig = px.pie(
                df,
                names="Segment"
            )

            st.plotly_chart(fig)


            col1,col2,col3 = st.columns(3)

            col1.metric(
                "Min Price",
                round(predictions.min(),2)
            )

            col2.metric(
                "Average Price",
                round(predictions.mean(),2)
            )

            col3.metric(
                "Max Price",
                round(predictions.max(),2)
            )

        except Exception as e:

            st.error(str(e))
