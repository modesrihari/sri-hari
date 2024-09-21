import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the models and scaler
linear_model = joblib.load('linear_regression_model.joblib')
ridge_model = joblib.load('ridge_regression_model.joblib')
scaler = joblib.load('scaler.joblib')

# Function to make predictions
def predict_calories(input_data, model_type):
    if model_type == 'Linear Regression':
        prediction = linear_model.predict(input_data)
    else:
        prediction = ridge_model.predict(input_data)
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
# Adjust this part based on your training dataset's features
expected_columns = ['TotalSteps', 'TotalDistance', 'Calories', 'SedentaryMinutes',
                    'LightlyActiveMinutes', 'FairlyActiveMinutes', 'VeryActiveMinutes', 'ActivityLevel_Moderate', 'ActivityLevel_High']

# Add missing columns with default value 0
for col in expected_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[expected_columns]

# Select model type for prediction
model_type = st.sidebar.selectbox("Select Model Type", ["Linear Regression", "Ridge Regression"])

# Predict button
if st.sidebar.button("Predict"):
    # Standardize input data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = predict_calories(input_data_scaled, model_type)
    st.write(f"Predicted Calories: {prediction[0]}")
