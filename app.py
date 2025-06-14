import streamlit as st

import streamlit as st
import pandas as pd
import joblib

# Load your trained model (make sure delivery_model.pkl is in the same directory)
model = joblib.load("voting_model.pkl")

st.title("🚚 Timelytics: Predict Order Delivery Time")

# User Inputs
category = st.selectbox("Select Product Category", ["Electronics", "Clothing", "Groceries", "Books"])
location = st.text_input("Enter Customer Location (e.g., City or Zip Code)")
shipping_method = st.selectbox("Select Shipping Method", ["Standard", "Express", "Same-Day"])

# Optional: Add other inputs your model needs (e.g., distance, weight)
# For simplicity, let’s assume the model can handle text features or they are encoded inside

# Prediction trigger
if st.button("Predict Delivery Time"):
    input_data = pd.DataFrame({
        "product_category": [category],
        "customer_location": [location],
        "shipping_method": [shipping_method]
    })

    try:
        prediction = model.predict(input_data)[0]
        st.success(f"📦 Estimated Delivery Time: {prediction} day(s)")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

