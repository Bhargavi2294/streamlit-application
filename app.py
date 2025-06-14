import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline

# --- Train a mini model with dummy data ---
data = pd.DataFrame({
    "product_category": ["Electronics", "Clothing", "Books"],
    "customer_location": ["Porbandar", "Delhi", "Chennai"],
    "shipping_method": ["Express", "Standard", "Same-Day"],
    "delivery_time": [2, 5, 1]
})

X = data.drop("delivery_time", axis=1)
y = data["delivery_time"]

model = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"),
    RandomForestRegressor()
)
model.fit(X, y)

# --- Streamlit UI ---
st.title("🚛 Timelytics ")

category = st.selectbox("Product Category", X['product_category'].unique())
location = st.text_input("Customer Location")
shipping = st.selectbox("Shipping Method", X['shipping_method'].unique())

if st.button("Predict Delivery Time"):
    input_data = pd.DataFrame({
        "product_category": [category],
        "customer_location": [location],
        "shipping_method": [shipping]
    })
    prediction = model.predict(input_data)[0]
    st.success(f"🕒 Estimated Delivery Time: {prediction:.1f} day(s)")
