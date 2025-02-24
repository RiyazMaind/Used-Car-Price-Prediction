import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained XGBoost model
best_xgb = joblib.load("best_model_xgb.pkl")

# Define expected features (as per the trained model)
expected_features = [
    "Year", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats",
    "Fuel_Type_Diesel", "Fuel_Type_LPG", "Fuel_Type_Petrol",
    "Transmission_Manual", "Owner_Type_Fourth & Above",
    "Owner_Type_Second", "Owner_Type_Third"
]

# Streamlit UI
st.title("Used Car Price Prediction App")

# User Inputs
year = st.number_input("Year of Manufacture", min_value=2000, max_value=2024, step=1, value=2015)
kilometers_driven = st.number_input("Kilometers Driven", min_value=0, step=1000, value=50000)
mileage = st.number_input("Mileage (kmpl)", min_value=0.0, step=0.1, value=18.0)
engine = st.number_input("Engine Capacity (cc)", min_value=600, max_value=5000, step=10, value=1500)
power = st.number_input("Power (bhp)", min_value=30, max_value=500, step=1, value=100)
seats = st.number_input("Number of Seats", min_value=2, max_value=9, step=1, value=5)

fuel_type = st.selectbox("Fuel Type", ["Diesel", "LPG", "Petrol"])
transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
owner_type = st.selectbox("Owner Type", ["Second", "Third", "Fourth & Above"])

# One-Hot Encoding
fuel_type_diesel = 1 if fuel_type == "Diesel" else 0
fuel_type_lpg = 1 if fuel_type == "LPG" else 0
fuel_type_petrol = 1 if fuel_type == "Petrol" else 0

transmission_manual = 1 if transmission == "Manual" else 0

owner_second = 1 if owner_type == "Second" else 0
owner_third = 1 if owner_type == "Third" else 0
owner_fourth = 1 if owner_type == "Fourth & Above" else 0

# Create input dictionary
input_dict = {
    "Year": year, "Kilometers_Driven": kilometers_driven, "Mileage": mileage,
    "Engine": engine, "Power": power, "Seats": seats,
    "Fuel_Type_Diesel": fuel_type_diesel, "Fuel_Type_LPG": fuel_type_lpg,
    "Fuel_Type_Petrol": fuel_type_petrol, "Transmission_Manual": transmission_manual,
    "Owner_Type_Fourth & Above": owner_fourth, "Owner_Type_Second": owner_second,
    "Owner_Type_Third": owner_third
}

# Ensure all expected features exist (Fix Feature Shape Mismatch)
for feature in best_xgb.feature_names_in_:
    if feature not in input_dict:
        input_dict[feature] = 0  # Default missing features to 0

# Convert input dictionary to NumPy array
input_data = np.array([input_dict[feature] for feature in best_xgb.feature_names_in_]).reshape(1, -1)

# Display Final Input Shape for Debugging
st.write(f"🔍 Final input shape: {input_data.shape}")

# Predict Price Button
if st.button("Predict Price"):
    prediction = best_xgb.predict(input_data)[0]
    st.success(f"Estimated Car Price: ₹ {prediction:,.2f} Lakh")

    # Feature Importance Display
    feature_importance = pd.DataFrame(
        {"Feature": best_xgb.feature_names_in_, "Importance": best_xgb.feature_importances_}
    ).sort_values(by="Importance", ascending=False)

    # Display Model Accuracy (R² score)
    model_r2 = 0.986  # As per your earlier message
    st.write(f"**Model Accuracy (R² Score):** {model_r2:.3f}")

