import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import time

# Set page configuration
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for responsiveness and style
st.markdown("""
    <style>
    :root {
        --primary: #FF7F50;
        --secondary: #1E90FF;
        --dark: #2c3e50;
        --light: #f8f9fa;
        --success: #28a745;
    }
    
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-image: linear-gradient(rgba(255,255,255,0.9), rgba(255,255,255,0.9)), 
                          url('https://cdn.pixabay.com/photo/2017/01/14/12/59/construction-1975174_1280.jpg');
        background-size: cover;
        background-attachment: fixed;
        padding: 10px;
    }
    .header {
        text-align: center;
        padding: 20px 10px;
        color: var(--dark);
        font-family: 'Arial Black', sans-serif;
        font-size: 36px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }
    .description {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
        padding: 0 10px;
        color: var(--dark);
    }
    .stButton > button {
        background-color: var(--primary);
        color: white;
        border-radius: 12px;
        padding: 10px 20px;
        font-size: 18px;
        transition: transform 0.2s, background-color 0.3s;
        width: 100%;
        border: none;
    }
    .stButton > button:hover {
        background-color: #FF4500;
        transform: scale(1.05);
    }
    .prediction-card {
        background-color: rgba(209, 231, 221, 0.9);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        animation: fadeIn 1s;
        margin-top: 30px;
        color: var(--dark);
        border-left: 5px solid var(--success);
    }
    .input-card {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    .model-card {
        background-color: rgba(248, 249, 250, 0.9);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 30px;
        color: var(--dark);
    }
    .social-icons a {
        color: var(--dark);
        margin: 0 10px;
        font-size: 24px;
        transition: color 0.3s;
    }
    .social-icons a:hover {
        color: var(--primary);
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

# Font Awesome for icons
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
""", unsafe_allow_html=True)

# Load dataset with error handling
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('concrete.csv')
        return data
    except FileNotFoundError:
        st.error("Error: The dataset file 'concrete.csv' was not found. Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {str(e)}")
        st.stop()

data = load_data()

# Train model with performance metrics
def train_model(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Calculate metrics
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, mae, r2

model, mae, r2 = train_model(data)

# Sidebar with additional information
with st.sidebar:
    st.markdown("## ℹ️ About This App")
    st.markdown("""
        This app predicts the **compressive strength** of concrete based on its composition.
        
        The model is trained on historical data using Linear Regression.
        
        **Model Performance:**
        - R² Score: {:.2f}
        - Mean Absolute Error: {:.2f} MPa
    """.format(r2, mae))
    
    st.markdown("---")
    st.markdown("### 📊 Data Statistics")
    st.dataframe(data.describe().style.format("{:.2f}"))
    
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
        1. Adjust the input parameters using the sliders
        2. Click the **Predict Strength** button
        3. View your predicted compressive strength
    """)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    # Header Section
    st.markdown("<h1 class='header'>🏗️ Concrete Strength Predictor</h1>", unsafe_allow_html=True)
    st.markdown("<div class='description'>👷 A Machine Learning Tool for Civil Engineers to Predict Compressive Strength of Concrete</div>", unsafe_allow_html=True)

    # Image with caption
    st.image("https://cdn.pixabay.com/photo/2016/11/22/07/09/cement-1846312_1280.jpg", 
             use_container_width=True, 
             caption="Civil Engineering in Action")

with col2:
    # Input Fields inside a card
    with st.container():
        st.markdown("### 📋 Concrete Mix Design Parameters")
        with st.form("input_form"):
            cement = st.slider('🧱 Cement (kg/m³)', min_value=0.0, max_value=1000.0, value=540.0, step=1.0)
            slag = st.slider('⛏️ Blast Furnace Slag (kg/m³)', min_value=0.0, max_value=400.0, value=0.0, step=1.0)
            fly_ash = st.slider('🌫️ Fly Ash (kg/m³)', min_value=0.0, max_value=200.0, value=0.0, step=1.0)
            water = st.slider('💧 Water (kg/m³)', min_value=0.0, max_value=300.0, value=162.0, step=1.0)
            superplasticizer = st.slider('🧪 Superplasticizer (kg/m³)', min_value=0.0, max_value=30.0, value=2.5, step=0.1)
            coarse_agg = st.slider('🪨 Coarse Aggregate (kg/m³)', min_value=0.0, max_value=1200.0, value=1040.0, step=1.0)
            fine_agg = st.slider('🏖️ Fine Aggregate (kg/m³)', min_value=0.0, max_value=1000.0, value=676.0, step=1.0)
            age = st.slider('📅 Age (days)', min_value=1, max_value=365, value=28, step=1)
            
            submitted = st.form_submit_button('🚀 Predict Strength')

# Prediction and Results
if submitted:
    with st.spinner('🔍 Analyzing your concrete mix...'):
        time.sleep(1)  # Simulate processing time
        
        input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
        prediction = model.predict(input_data)
        
        # Display prediction with animation
        st.markdown(f"""
            <div class='prediction-card'>
                <h2>🧱 Predicted Compressive Strength</h2>
                <h1 style="color: var(--primary);">{prediction[0]:.2f} MPa</h1>
                <p>Based on your input parameters at {age} days</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Additional interpretation
        with st.expander("💡 Interpretation"):
            if prediction[0] < 20:
                st.warning("This mix design results in relatively low strength concrete. Consider increasing cement content or reducing water content.")
            elif prediction[0] > 60:
                st.success("This mix design results in high-strength concrete. Suitable for structural applications.")
            else:
                st.info("This mix design results in standard strength concrete. Appropriate for most general construction purposes.")
            
            st.markdown("""
                **Typical Strength Ranges:**
                - Residential slabs: 20-25 MPa
                - Structural beams/columns: 25-40 MPa
                - High-rise buildings: 40-60 MPa
                - Special applications: 60+ MPa
            """)

# Footer
st.markdown("---")
st.markdown("""
    <div class='footer'>
        <p>&copy; 2025 Concrete Strength Predictor | All rights reserved</p>
        <p><strong>Developed with ❤️ by Irfan Ullah Khan</strong></p>
        <div class="social-icons">
            <a href="https://github.com/programmarself" target="_blank" title="GitHub"><i class="fab fa-github"></i></a>
            <a href="https://www.linkedin.com/in/iukhan/" target="_blank" title="LinkedIn"><i class="fab fa-linkedin"></i></a>
            <a href="https://programmarself.github.io/My_Portfolio/" target="_blank" title="Portfolio"><i class="fa fa-briefcase"></i></a>
            <a href="mailto:your-email@example.com" title="Email"><i class="fas fa-envelope"></i></a>
        </div>
    </div>
""", unsafe_allow_html=True)
