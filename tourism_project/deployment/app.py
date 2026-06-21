import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_REPO_ID = "Nikidsouza23/visit-with-us-tourism-model"
MODEL_FILENAME = "best_tourism_model.joblib"

@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(model_path)

model = load_model()

st.set_page_config(page_title="Visit with Us Predictor", layout="centered")
st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer is likely to purchase the package before contacting them.")

age = st.number_input("Age", min_value=18, max_value=100, value=35)
typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
numberofpersonvisiting = st.number_input("Number of Person Visiting", min_value=1, max_value=20, value=2)
preferredpropertystar = st.number_input("Preferred Property Star", min_value=1.0, max_value=5.0, value=3.0)
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
numberoftrips = st.number_input("Number of Trips", min_value=0.0, max_value=20.0, value=2.0)
passport = st.selectbox("Passport", [0, 1])
owncar = st.selectbox("Own Car", [0, 1])
numberofchildrenvisiting = st.number_input("Number of Children Visiting", min_value=0.0, max_value=10.0, value=0.0)
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
monthlyincome = st.number_input("Monthly Income", min_value=1000.0, value=25000.0)
pitchsatisfactionscore = st.number_input("Pitch Satisfaction Score", min_value=1.0, max_value=5.0, value=3.0)
productpitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
numberoffollowups = st.number_input("Number of Followups", min_value=0.0, max_value=10.0, value=3.0)
durationofpitch = st.number_input("Duration of Pitch", min_value=1.0, max_value=100.0, value=15.0)
citytier = st.selectbox("City Tier", [1, 2, 3])

input_df = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "ProductPitched": productpitched,
    "NumberOfFollowups": numberoffollowups,
    "DurationOfPitch": durationofpitch
}])

if st.button("Predict"):
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if pred == 1:
        st.success("Prediction: Likely to Purchase")
    else:
        st.warning("Prediction: Not Likely to Purchase")

    st.info(f"Predicted purchase probability: {prob:.2%}")
    st.dataframe(input_df)
