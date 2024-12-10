#Using ipynb , I have created pkl model.
#Dr.Arun Anoop M., Associate Professor, Dept. of CSE, VCET Puttur.
import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained Logistic Regression model
model = joblib.load('C:/Users/arunanoopm/Downloads/logistic_model.pkl')

# Load the crop recommendation data
crop_data = pd.read_csv('C:/Users/arunanoopm/Downloads/Crop_recommendation.csv')


# Display the title of the app
st.title("Crop Recommendation System")

# Display the crop recommendation dataset
st.write("Crop Recommendation Dataset", crop_data.head())

# Create input fields for the user to enter their data

Nitrogen = st.number_input("Nitrogen")
Phosphorus = st.number_input("Phosphorus")
Potassium = st.number_input("Potassium")
Temperature = st.number_input("Temperature")
Humidity = st.number_input("Humidity")
pH_Value = st.number_input("pH_Value")
Rainfall = st.number_input("Rainfall")

# Feature order: Temperature, Humidity, pH, Rainfall
user_input = pd.DataFrame({
    'Nitrogen': [Nitrogen],
    'Phosphorus': [Phosphorus],
    'Potassium': [Potassium],
    'Temperature': [Temperature],
    'Humidity': [Humidity],
    'pH_Value': [pH_Value],
    'Rainfall': [Rainfall]
})

# Make a prediction when the user clicks the 'Recommend' button
if st.button('Recommend Crop'):
    # Predict the crop using the model
    prediction = model.predict(user_input)
    
    # Get the name of the crop from the prediction
    recommended_crop = prediction[0]

    # Display the recommendation
    st.write(f"The recommended crop for the given conditions is: {recommended_crop}")
    
    if 'Crop' in crop_data.columns:
        # If the column exists, attempt to get information about the crop
        crop_info = crop_data[crop_data['Crop'] == recommended_crop]
        st.write("More details about the recommended crop:", crop_info)
    else:
        # If the column doesn't exist, show an appropriate message
        st.write(" ")