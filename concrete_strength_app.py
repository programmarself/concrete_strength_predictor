import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Set page configuration
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-image: url('https://cdn.pixabay.com/photo/2017/01/14/12/59/construction-1975174_1280.jpg');
        background-size: cover;
        background-attachment: fixed;
        padding: 10px;
    }
    .header {
        text-align: center;
        padding: 20px 10px;
        color: #FF7F50;
        font-family: 'Arial Black', sans-serif;
        font-size: 36px;
    }
    .description {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
        padding: 0 10px;
    }
    .stButton > button {
        background-color: #FF7F50;
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        transition: transform 0.2s, background-color 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #FF4500;
        transform: scale(1.05);
    }
    .prediction-card {
        background-color: #d1e7dd;
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.2);
        animation: fadeIn 1s;
        margin-top: 30px;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    /* Mobile adjustments */
    @media only screen and (max-width: 768px) {
        .header {
            font-size: 28px;
        }
        .description {
            font-size: 16px;
        }
        .stButton > button {
            font-size: 16px;
            padding: 10px;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('concrete.csv')

data = load_data()

def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(data)

# Header Section
st.markdown("<h1 class='header'>🏗️ Concrete Strength Predictor</h1>", unsafe_allow_html=True)
st.markdown("<div class='description'>👷 A Machine Learning Tool for Civil Engineers to Predict Compressive Strength of Concrete</div>", unsafe_allow_html=True)

st.image("https://cdn.pixabay.com/photo/2016/11/22/07/09/cement-1846312_1280.jpg", use_container_width=True, caption="Civil Engineering in Action")

st.markdown("### 📋 Enter Concrete Mix Details:")

# Input Fields inside a container for responsiveness
with st.container():
    cement = st.number_input('🧱 Cement (kg/m³)', min_value=0.0, max_value=1000.0, value=540.0)
    slag = st.number_input('⛏️ Blast Furnace Slag (kg/m³)', min_value=0.0, max_value=400.0, value=0.0)
    fly_ash = st.number_input('🌫️ Fly Ash (kg/m³)', min_value=0.0, max_value=200.0, value=0.0)
    water = st.number_input('💧 Water (kg/m³)', min_value=0.0, max_value=300.0, value=162.0)
    superplasticizer = st.number_input('🧪 Superplasticizer (kg/m³)', min_value=0.0, max_value=30.0, value=2.5)
    coarse_agg = st.number_input('🪨 Coarse Aggregate (kg/m³)', min_value=0.0, max_value=1200.0, value=1040.0)
    fine_agg = st.number_input('🏖️ Fine Aggregate (kg/m³)', min_value=0.0, max_value=1000.0, value=676.0)
    age = st.number_input('📅 Age (days)', min_value=1, max_value=365, value=28)

# Prediction Button
if st.button('🚀 Predict Strength'):
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    prediction = model.predict(input_data)
    st.markdown(f"""
        <div class='prediction-card'>
            <h2>🧱 Predicted Compressive Strength:</h2>
            <h1>{prediction[0]:.2f} MPa</h1>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
     <div style='text-align: center; font-size: 16px;'>
      <p>&copy; 2025 Concrete Strength Predictor | All rights reserved</p>
      <p><strong>Developed with ❤️ by Irfan Ullah Khan</strong></p>
      <div class="social-links">
          <a href="https://github.com/programmarself" target="_blank"><i class="fab fa-github"></i></a>
          <a href="https://www.linkedin.com/in/iukhan/" target="_blank"><i class="fab fa-linkedin"></i></a>
          <a href="https://programmarself.github.io/My_Portfolio/" target="_blank"><i class="fa fa-briefcase"></i></a>
        </div>
    </div>
""", unsafe_allow_html=True)
