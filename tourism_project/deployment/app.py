import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

# Load the model from Hugging Face
model_path = hf_hub_download(
    repo_id="arkac/tourism-package-purchase-prediction-model", 
    filename="best_tourism_package_purchase_prediction_model.joblib"
)
model = joblib.load(model_path)

st.title("Tourism Package Purchase Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

# Input features
age = st.number_input("Age", min_value=18, max_value=61, value=30)
type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=5.0, max_value=127.0, value=10.0)
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
# Note: dataset contained a typo 'Fe Male' which is normalized during preprocessing to 'Female'
gender = st.selectbox("Gender", ["Female", "Male"])
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=5, value=2)
number_of_followups = st.number_input("Number of Followups", min_value=1.0, max_value=6.0, value=3.0)
product_pitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
preferred_property_star = st.number_input("Preferred Property Star", min_value=3.0, max_value=5.0, value=3.0)
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
number_of_trips = st.number_input("Number of Trips", min_value=1.0, max_value=22.0, value=2.0)
passport = st.selectbox("Passport", [0, 1])
pitch_satisfaction_score = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
own_car = st.selectbox("Own Car", [0, 1])
number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=3.0, value=0.0)
designation = st.selectbox("Designation", ["Manager", "Senior Manager", "Executive", "VP", "AVP"])
monthly_income = st.number_input("Monthly Income", min_value=1000.0, max_value=98678.0, value=20000.0)

# Create dataframe with raw categorical values (no manual encoding needed)
# The preprocessing pipeline in the model will handle scaling and one-hot encoding
input_data = pd.DataFrame({
    "Age": [age],
    "TypeofContact": [type_of_contact],
    "CityTier": [city_tier],
    "DurationOfPitch": [duration_of_pitch],
    "Occupation": [occupation],
    "Gender": [gender],
    "NumberOfPersonVisiting": [number_of_person_visiting],
    "NumberOfFollowups": [number_of_followups],
    "ProductPitched": [product_pitched],
    "PreferredPropertyStar": [preferred_property_star],
    "MaritalStatus": [marital_status],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "PitchSatisfactionScore": [pitch_satisfaction_score],
    "OwnCar": [own_car],
    "NumberOfChildrenVisiting": [number_of_children_visiting],
    "Designation": [designation],
    "MonthlyIncome": [monthly_income]
})

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("The customer is likely to purchase the Wellness Tourism Package.")
    else:
        st.error("The customer is not likely to purchase the Wellness Tourism Package.")
