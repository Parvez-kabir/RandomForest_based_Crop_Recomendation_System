import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import base64

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
# Title & Intro
# -----------------------------
st.markdown("<h1 style='text-align: center; color: darkgreen;'>🌾 Advanced Crop Recommendation System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Predict the most suitable crop based on soil nutrients and weather conditions.</p>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

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
# Crop Images Dictionary
# -----------------------------
crop_images = {
    "Apple": "images/apple.jpg",
    "Banana": "images/banana.jpg",
    "Blackgram": "images/blackgram.jpg",
    "ChickPea": "images/ChickPea.jpg",
    "Coconut": "images/coconut.jpg",
    "Coffee": "images/coffee.jpg",
    "Cotton": "images/cotton.jpg",
    "grape": "images/grape.jpg",
    "Jute": "images/jute.jpg",
    "Kidneybeans": "images/kidneybeans.jpg",
    "Lentil": "images/lentil.jpg",
    "Maize": "images/maize.jpg",
    "Mango": "images/mango.jpg",
    "Mothbean": "images/mothbean.jpg",
    "Muskmelon": "images/Muskmelon.jpg",
    "Papaya": "images/papaya.jpg",
    "PigeonPea": "images/pigeonPea.jpg",
    "Promeganate": "images/promeganete.jpg",
    "Rice": "images/rice.jpg",
    "Watermelon": "images/watermelon.jpg",
    "Mungbean": "images/mungbean.jpg"
}

# -----------------------------
# Convert image to base64
# -----------------------------
def img_to_bytes(img_path):
    with open(img_path, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode()

# -----------------------------
# Prediction
# -----------------------------
if st.sidebar.button("Predict Crop"):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    
    # Predict crop
    pred = model.predict(input_data)
    pred_crop = le.inverse_transform(pred)[0]
    
    # Predict probabilities
    prob = model.predict_proba(input_data)[0]
    prob_df = pd.DataFrame({
        "Crop": le.classes_,
        "Confidence (%)": np.round(prob*100, 2)
    })
    prob_df = prob_df[prob_df['Confidence (%)'] > 0].sort_values(by="Confidence (%)", ascending=False)

    # -----------------------------
    # Recommended Crop Card
    # -----------------------------
    if pred_crop in crop_images:
        img_bytes = img_to_bytes(crop_images[pred_crop])
        st.markdown(f"""
            <div style="
                border-radius: 15px;
                background-color: #d4edda;
                padding: 20px;
                text-align: center;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
                margin: 20px auto;
                max-width: 500px;
            ">
                <h3 style='color: #155724; margin-bottom:15px;'>🌱 Recommended Crop</h3>
                <img src='data:image/jpeg;base64,{img_bytes}' style='max-width: 90%; border-radius:10px;'/>
                <h2 style='color: #155724; margin-top:15px;'>{pred_crop}</h2>
            </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # Top 3 Predictions Modern Cards
    # -----------------------------
    st.subheader("🌟 Top 3 Crop Predictions")
    top3 = prob_df.head(3)
    top3_cols = st.columns(3)
    for idx, (_, row) in enumerate(top3.iterrows()):
        crop_name = row["Crop"]
        conf = row["Confidence (%)"]
        if crop_name in crop_images:
            img_bytes = img_to_bytes(crop_images[crop_name])
            with top3_cols[idx]:
                st.markdown(f"""
                    <div style="
                        border-radius: 15px;
                        background-color: #e0f7fa;
                        padding: 10px;
                        text-align: center;
                        box-shadow: 0 6px 15px rgba(0,0,0,0.2);
                        margin-bottom: 10px;
                    ">
                        <img src='data:image/jpeg;base64,{img_bytes}' style='width:100px; height:100px; border-radius:10px;'/><br>
                        <b>{crop_name}</b><br>
                        <span>Confidence: {conf}%</span>
                    </div>
                """, unsafe_allow_html=True)

    # -----------------------------
    # Histogram / Bar Chart with images
    # -----------------------------
    st.subheader("📊 Prediction Confidence Histogram")
    fig = go.Figure()
    for _, row in prob_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Crop']],
            y=[row['Confidence (%)']],
            marker_color='mediumseagreen',
            name=row['Crop']
        ))
        if row['Crop'] in crop_images:
            img_bytes = img_to_bytes(crop_images[row['Crop']])
            fig.add_layout_image(
                dict(
                    source=f"data:image/jpeg;base64,{img_bytes}",
                    x=row['Crop'],
                    y=row['Confidence (%)'] + 5,  # slightly above bar
                    xref="x",
                    yref="y",
                    sizex=0.8,
                    sizey=20,
                    xanchor="center",
                    yanchor="bottom",
                    sizing="contain",
                    layer="above"
                )
            )
    fig.update_layout(
        yaxis=dict(title='Confidence (%)', range=[0, 110]),
        xaxis=dict(title='Crops'),
        height=600,
        showlegend=False,
        margin=dict(t=40)
    )
    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------
    # Detailed Confidence Table
    # -----------------------------
    with st.expander("See Detailed Confidence Table"):
        st.dataframe(prob_df.reset_index(drop=True))
