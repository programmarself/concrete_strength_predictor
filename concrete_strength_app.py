import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('concrete.csv')
    return data

data = load_data()

def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model(data)

# Custom CSS for styling and animations
st.markdown("""
    <style>
    body {
        background-color: #e6e6e6;
    }
    .main {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }
    h1, h2 {
        color: #FF7F50;
        text-align: center;
        font-family: 'Arial Black', Gadget, sans-serif;
    }
    .stButton > button {
        background-color: #FF7F50;
        color: white;
        border-radius: 12px;
        transition: transform 0.2s, background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #FF4500;
        transform: scale(1.05);
    }
    .prediction-card {
        background-color: #d1e7dd;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.2);
        text-align: center;
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<h1>🏗️ Concrete Strength Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h2>🔧 Powered by Machine Learning</h2>", unsafe_allow_html=True)
st.image("https://cdn.pixabay.com/photo/2017/01/14/12/59/construction-1975174_1280.jpg", use_column_width=True, caption="Civil Engineering in Action")

st.markdown("### 📋 Enter Concrete Mix Details:")

# Input Section
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
        <div class="prediction-card">
            <h2>🧱 Predicted Compressive Strength:</h2>
            <h1>{prediction[0]:.2f} MPa</h1>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center;'>
        <p>Developed with ❤️ by Irfan Ullah Khan</p>
        <p><a href='https://github.com/programmarself/concrete_strength_predictor' target='_blank'>View on GitHub</a></p>
    </div>
""", unsafe_allow_html=True)
