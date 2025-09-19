

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# -----------------------------
# Load Model and Encoder
# -----------------------------
model = joblib.load("crop_model.pkl")
le = joblib.load("label_encoder.pkl")

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="🌾 Advanced Crop Recommendation",
    page_icon="🌱",
    layout="wide"
)

# -----------------------------
# Background CSS
# -----------------------------
def set_bg_image(image_file):
    import base64
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Uncomment and set your image path
# set_bg_image("background.jpg")

# -----------------------------
# Title & Intro
# -----------------------------
st.title("🌾 Advanced Crop Recommendation System")
st.markdown("""
Predict the most suitable crop based on soil and weather conditions.  
This app is designed for farmers, agronomists, and agricultural enthusiasts.
""")

# -----------------------------
# Sidebar Input
# -----------------------------
st.sidebar.header("Enter Soil & Weather Details")

N = st.sidebar.slider("Nitrogen (N)", 0, 140, 90)
P = st.sidebar.slider("Phosphorus (P)", 0, 140, 42)
K = st.sidebar.slider("Potassium (K)", 0, 140, 43)
temperature = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 20.88)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 82.0)
ph = st.sidebar.slider("pH Value", 0.0, 14.0, 6.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 500.0, 202.9)

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Predict
    pred = model.predict(input_data)
    pred_crop = le.inverse_transform(pred)[0]
    
    # Predict probabilities
    prob = model.predict_proba(input_data)[0]
    prob_df = pd.DataFrame({
        "Crop": le.classes_,
        "Confidence (%)": np.round(prob*100, 2)
    }).sort_values(by="Confidence (%)", ascending=False)
    
    # -----------------------------
    # Display Recommended Crop
    # -----------------------------
    st.success(f"🌱 Recommended Crop: **{pred_crop}**")
    
    # -----------------------------
    # Bar Chart
    # -----------------------------
    st.subheader("Prediction Confidence (Bar Chart)")
    st.bar_chart(data=prob_df.set_index('Crop'))
    
    # -----------------------------
    # Pie Chart
    # -----------------------------
    st.subheader("Prediction Confidence (Pie Chart)")
    fig_pie = px.pie(prob_df, names='Crop', values='Confidence (%)',
                     color_discrete_sequence=px.colors.sequential.Viridis)
    st.plotly_chart(fig_pie)
    
    # -----------------------------
    # Top 3 Crop Gauge / Progress
    # -----------------------------
    st.subheader("Top 3 Crop Probabilities")
    top3 = prob_df.head(3)
    for i, row in top3.iterrows():
        st.write(f"**{row['Crop']}**")
        st.progress(int(row['Confidence (%)']))
    
    # -----------------------------
    # Detailed Table
    # -----------------------------
    with st.expander("See Detailed Confidence Table"):
        st.dataframe(prob_df.reset_index(drop=True))


