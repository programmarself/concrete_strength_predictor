import streamlit as st

st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to:", ["Home", "Predict Strength", "About"])

if selection == "Home":
    st.image("https://cdn.pixabay.com/photo/2017/01/14/12/59/construction-1975174_1280.jpg", use_column_width=True)
    st.title("🏗️ Concrete Strength Predictor")
    st.markdown("### 👷 A Machine Learning Powered Tool for Civil Engineers")
    st.markdown("#### 🎯 Predict the **compressive strength of concrete** using input mix proportions.")
    st.image("https://cdn.pixabay.com/photo/2016/11/22/07/09/cement-1846312_1280.jpg", use_column_width=True, caption="Civil Engineering in Action")

elif selection == "Predict Strength":
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

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

    st.title("🧮 Predict Concrete Strength")

    cement = st.number_input('🧱 Cement (kg/m³)', min_value=0.0, max_value=1000.0, value=540.0)
    slag = st.number_input('⛏️ Blast Furnace Slag (kg/m³)', min_value=0.0, max_value=400.0, value=0.0)
    fly_ash = st.number_input('🌫️ Fly Ash (kg/m³)', min_value=0.0, max_value=200.0, value=0.0)
    water = st.number_input('💧 Water (kg/m³)', min_value=0.0, max_value=300.0, value=162.0)
    superplasticizer = st.number_input('🧪 Superplasticizer (kg/m³)', min_value=0.0, max_value=30.0, value=2.5)
    coarse_agg = st.number_input('🪨 Coarse Aggregate (kg/m³)', min_value=0.0, max_value=1200.0, value=1040.0)
    fine_agg = st.number_input('🏖️ Fine Aggregate (kg/m³)', min_value=0.0, max_value=1000.0, value=676.0)
    age = st.number_input('📅 Age (days)', min_value=1, max_value=365, value=28)

    if st.button('🚀 Predict Strength'):
        input_data = np.array([[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]])
        prediction = model.predict(input_data)
        st.markdown(f"""
            <div style='background-color: #d1e7dd; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); animation: fadeIn 1s;'>
                <h2>🧱 Predicted Compressive Strength:</h2>
                <h1>{prediction[0]:.2f} MPa</h1>
            </div>
            <style>
            @keyframes fadeIn {{
                from {{opacity: 0;}}
                to {{opacity: 1;}}
            }}
            </style>
        """, unsafe_allow_html=True)

elif selection == "About":
    st.title("ℹ️ About This App")
    st.image("https://cdn.pixabay.com/photo/2017/01/20/00/30/civil-engineer-1995857_1280.jpg", use_column_width=True)
    st.markdown("""
        This application is designed for **civil engineers** to easily predict the compressive strength of concrete based on the mixture inputs.
        \nIt uses a **machine learning model (Linear Regression)** trained on a real-world dataset.
        \nThe goal is to make **concrete strength prediction fast, reliable, and accessible.**
    """)
    st.markdown("""
        ---
        Developed with ❤️ by **Irfan Ullah Khan**
        \n🔗 [GitHub Repository](https://github.com/programmarself/concrete_strength_predictor)
    """)
