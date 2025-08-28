import streamlit as st
import joblib
import pandas as pd

# --------------------------
# Load the trained pipeline (preprocessor + regressor)
# --------------------------
regressor = joblib.load("notebook/linear_model.pkl")

# --------------------------
# Title
# --------------------------
st.title("📊 Student Score Prediction Web App")
st.markdown("Enter student details to predict the **Math Score** using the trained Linear Regression model.")

# --------------------------
# Input fields
# --------------------------
gender = st.selectbox("Gender", ["male", "female"])
race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parental_level_of_education = st.selectbox(
    "Parental Level of Education",
    ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
)
lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])

reading_score = st.number_input("Reading Score", min_value=0, max_value=100, step=1)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, step=1)

# --------------------------
# Predict button
# --------------------------
if st.button("Predict Math Score"):

    # Input as DataFrame (raw categorical + numeric)
    input_df = pd.DataFrame({
        "gender": [gender],
        "race_ethnicity": [race_ethnicity],
        "parental_level_of_education": [parental_level_of_education],
        "lunch": [lunch],
        "test_preparation_course": [test_preparation_course],
        "reading_score": [reading_score],
        "writing_score": [writing_score]
    })

    # ✅ No manual preprocessing needed, pipeline handles it
    prediction = regressor.predict(input_df)

    st.success(f"🎯 Predicted Math Score: {prediction[0]:.2f}")
