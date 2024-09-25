import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
import base64


with open("le_town.pkl","rb") as files:
  le_town=pickle.load(files)

with open("le_flat.pkl","rb") as files:
  le_flat=pickle.load(files)

with open("le_flat_model.pkl","rb") as files:
  le_flat_model=pickle.load(files)

with open('DecisionTreeRegressor.pkl','rb') as files:
    model_dt=pickle.load(files)



df=pd.read_csv("Singapore_Resale_Flat_Prices.csv")

st.set_page_config(page_title="Resale Price Prediction", page_icon="🏨",layout="wide")

st.markdown(
    """
    <h2 style='color: #4B3D24; font-size: 35px; text-align: center'>SINGAPORE FLAT RESALE PRICE PREDICTION</h2>
    """,
    unsafe_allow_html=True
)

def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
background_image_path = r'i2.jpg'
base64_image = get_base64_of_bin_file(background_image_path)
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/jpg;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    ;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)


placeholder = st.empty()

with placeholder.container():
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        town = st.selectbox("Select Town", le_town.classes_)
        flat_type = st.selectbox("Select Flat Type", le_flat.classes_)
        flat_model = st.selectbox("Select Flat Model", le_flat_model.classes_)
        floor_area_sqm = st.number_input(label='Floor Area (sqm)', min_value=10.0)
        remaining_lease = st.number_input("Remaining Lease in Years", min_value=0, max_value=99)
        resale_year = st.text_input(label="Year", max_chars=4)
        resale_month = st.text_input(label="Month", max_chars=2)
        lease_age = st.number_input(label="Lease Age (years)", min_value=0, max_value=99)
        block = st.number_input("Block", value=1.0)
        lease_commence_date = st.number_input("Lease Commence Date", value=2013.0)
        storey_low = st.number_input("Storey Low", value=1.0)
        storey_high = st.number_input("Storey High", value=3.0)

        block_log = np.log1p(block)
        lease_commence_date_log = np.log1p(lease_commence_date)
        storey_low_log = np.log1p(storey_low)
        storey_high_log = np.log1p(storey_high)

        input_data = pd.DataFrame({
            "town": [town],
            "flat_type": [flat_type],
            "floor_area_sqm": [floor_area_sqm],
            "flat_model": [flat_model],
            "remaining_lease": [remaining_lease],
            "resale_year": [resale_year],
            "resale_month": [resale_month],
            "lease_age": [lease_age],
            "block_log": [block_log],
            "lease_commence_date_log": [lease_commence_date_log],
            "storey_low_log": [storey_low_log],
            "storey_high_log": [storey_high_log]
        })

        input_data["town"] = le_town.transform(input_data["town"])
        input_data["flat_type"] = le_flat.transform(input_data["flat_type"])
        input_data["flat_model"] = le_flat_model.transform(input_data["flat_model"])

        if st.button("Predict Resale Price"):
            prediction = model_dt.predict(input_data)
            predicted_price = np.exp(prediction[0])  
            st.markdown(f"<h3 style='color: #36454F;'>The predicted Resale Price is: ${predicted_price:,.2f}</h3>", unsafe_allow_html=True)