import streamlit as st
import pandas as pd
import joblib

# Load the linear regression model
try:
    linear_model = joblib.load('linear_regression_model.joblib')
except FileNotFoundError as e:
    st.error("Model file not found. Please ensure it is in the same directory as the app.")
    st.stop()  # Stop the app if the file is not found

# Function to make predictions
def predict_calories(input_data):
    prediction = linear_model.predict(input_data)
    return prediction

# Streamlit app title
st.title("Calories Prediction App")

# Input features
st.sidebar.header("Input Features")
features = {}

# Example feature inputs based on common activity dataset columns
features['TotalSteps'] = st.sidebar.number_input("Total Steps", min_value=0)
features['TotalDistance'] = st.sidebar.number_input("Total Distance (miles)", min_value=0.0)
features['Calories'] = st.sidebar.number_input("Calories Consumed", min_value=0)
features['ActivityLevel'] = st.sidebar.selectbox("Activity Level", ["Low", "Moderate", "High"])
features['SedentaryMinutes'] = st.sidebar.number_input("Sedentary Minutes", min_value=0)
features['LightlyActiveMinutes'] = st.sidebar.number_input("Lightly Active Minutes", min_value=0)
features['FairlyActiveMinutes'] = st.sidebar.number_input("Fairly Active Minutes", min_value=0)
features['VeryActiveMinutes'] = st.sidebar.number_input("Very Active Minutes", min_value=0)

# Convert input features to DataFrame
input_data = pd.DataFrame([features])

# Encode categorical feature (ActivityLevel)
input_data = pd.get_dummies(input_data, columns=['ActivityLevel'], drop_first=True)

# Ensure input_data has the same columns as the training data
expected_columns = ['TotalSteps', 'TotalDistance', 'Calories', 'SedentaryMinutes',
                    'LightlyActiveMinutes', 'FairlyActiveMinutes', 'VeryActiveMinutes', 'ActivityLevel_Moderate', 'ActivityLevel_High']

# Add missing columns with default value 0
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_columns]

# Predict button
if st.sidebar.button("Predict"):
    # Make prediction
    prediction = predict_calories(input_data)
    st.write(f"Predicted Calories: {prediction[0]}")
