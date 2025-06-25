import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv('concrete.csv')  # Reading from local file in the repo
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

st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🏗️ Concrete Compressive Strength Predictor")
st.subheader("Civil Engineering | Machine Learning | Streamlit")
st.write("Enter the concrete mix details below to predict the compressive strength (MPa).")

cement = st.number_input('Cement (kg/m³)', min_value=0.0, max_value=1000.0, value=540.0)
slag = st.number_input('Blast Furnace Slag (kg/m³)', min_value=0.0, max_value=400.0, value=0.0)
fly_ash = st.number_input('Fly Ash (kg/m³)', min_value=0.0, max_value=200.0, value=0.0)
water = st.number_input('Water (kg/m³)', min_value=0.0, max_value=300.0, value=162.0)
superplasticizer = st.number_input('Superplasticizer (kg/m³)', min_value=0.0, max_value=30.0, value=2.5)
coarse_agg = st.number_input('Coarse Aggregate (kg/m³)', min_value=0.0, max_value=1200.0, value=1040.0)
fine_agg = st.number_input('Fine Aggregate (kg/m³)', min_value=0.0, max_value=1000.0, value=676.0)
age = st.number_input('Age (days)', min_value=1, max_value=365, value=28)

if st.button('Predict Strength'):
    input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
    prediction = model.predict(input_data)
    st.success(f"Predicted Compressive Strength: {prediction[0]:.2f} MPa")
