import streamlit as st
from streamlit_option_menu import option_menu

import Prediction
import Batch_Prediction



st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide"
)
st.image("Designer.png")

with st.sidebar:

    st.image("Designer.png")

    selected = option_menu(
        menu_title="Property Price Predictor",
        options=[
            "Home",
            "Price Prediction",
            "Batch Prediction"
        ],
        icons=[
            "house-fill",
            "currency-rupee",
            "file-earmark-spreadsheet"
        ],
        menu_icon="building",
        default_index=0
    )

if selected == "Home":
    
    st.write("")
    tab1,tab2,tab3 = st.tabs(
    ["Overview","Methodology","Author"]
    )

    with tab1:

        st.markdown("""
        ### Project Overview

        Predict residential property prices using
        Machine Learning models.

        Features:

        - Single Prediction
        - Batch Prediction
        - Model Analytics
        - Performance Dashboard
        """)

    with tab2:

        st.markdown("""
        ### Methodology

        1. Data Collection
        2. Data Cleaning
        3. EDA
        4. Feature Engineering
        5. Scaling
        6. Model Training
        7. Hyperparameter Tuning
        8. Deployment
        """)

    with tab3:

        st.markdown("""
        ### Author

        Brijesh Singh

        PG Diploma DSML

        Email:
        brijesh.singh@aol.com
        """)

elif selected == "Price Prediction":
    Prediction.show()

elif selected == "Batch Prediction":
    Batch_Prediction.show()




