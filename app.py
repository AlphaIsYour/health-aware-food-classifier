import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Food Recommendation System for Diabetes",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4788;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-safe {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .result-unsafe {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #dee2e6;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and preprocessor
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_diabetes_food_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please ensure you have run the training notebook first.")
        st.stop()

model, scaler, feature_names = load_model_and_scaler()

# Prediction function
def predict_diabetes_safety(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    input_scaled = pd.DataFrame(input_scaled, columns=feature_names)
    
    prediction_label = model.predict(input_scaled)[0]
    prediction_text = "Safe" if prediction_label == 1 else "Not Safe"
    probabilities = model.predict_proba(input_scaled)[0]
    confidence = probabilities[prediction_label]
    
    return {
        'prediction': prediction_text,
        'prediction_label': int(prediction_label),
        'confidence': float(confidence),
        'probabilities': {
            'Not Safe': float(probabilities[0]),
            'Safe': float(probabilities[1])
        }
    }

# Header
st.markdown('<p class="main-header">Food Recommendation System for Diabetes Patients</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Nutritional Classification Based on Machine Learning</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About This System")
    st.write("""
    This application uses machine learning to classify foods based on their safety for diabetes patients.
    
    **Classification Criteria:**
    - Sugar content
    - Carbohydrate levels
    - Total calories
    - Other nutritional factors
    
    **Model:** Random Forest Classifier
    """)
    
    st.divider()
    
    st.subheader("How to Use")
    st.write("""
    1. Select a food from the dropdown menu
    2. Review the nutritional information
    3. Click 'Analyze Food Safety' button
    4. View the prediction results
    """)

# Main content
tab1, tab2 = st.tabs(["Food Analysis", "Manual Input"])

with tab1:
    st.header("Select Food for Analysis")
    
    # Load dataset for food selection
    @st.cache_data
    def load_food_data():
        try:
            df = pd.read_csv('nilai-gizi.csv')
            # Clean numeric columns
            for col in feature_names:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except FileNotFoundError:
            st.error("Dataset not found. Please upload nilai-gizi.csv")
            return None
    
    df = load_food_data()
    
    if df is not None:
        # Food selection
        food_names = df['name'].dropna().unique().tolist()
        selected_food = st.selectbox(
            "Choose a food item:",
            options=food_names,
            index=0
        )
        
        # Get food data
        food_data = df[df['name'] == selected_food].iloc[0]
        
        # Display nutritional information
        st.subheader("Nutritional Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Serving Size", f"{food_data['serving_size']:.0f} g")
            st.metric("Energy", f"{food_data['energy_kcal']:.1f} kcal")
        
        with col2:
            st.metric("Protein", f"{food_data['protein_g']:.1f} g")
            st.metric("Carbohydrates", f"{food_data['carbohydrate_g']:.1f} g")
        
        with col3:
            st.metric("Fat", f"{food_data['fat_g']:.1f} g")
            st.metric("Sugar", f"{food_data['sugar_g']:.1f} g")
        
        with col4:
            st.metric("Sodium", f"{food_data['sodium_mg']:.1f} mg")
            st.metric("Fiber", f"{food_data['fiber_g']:.1f} g")
        
        st.divider()
        
        # Predict button
        if st.button("Analyze Food Safety", type="primary", use_container_width=True):
            # Prepare input
            input_data = {feature: food_data[feature] for feature in feature_names}
            
            # Make prediction
            with st.spinner("Analyzing nutritional data..."):
                result = predict_diabetes_safety(input_data)
            
            # Display results
            st.subheader("Analysis Results")
            
            if result['prediction'] == "Safe":
                st.markdown(f"""
                <div class="result-safe">
                    <h3 style="color: #28a745; margin-top: 0;">✓ SAFE FOR DIABETES PATIENTS</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 0;">
                        This food is recommended for people with diabetes based on its nutritional profile.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-unsafe">
                    <h3 style="color: #dc3545; margin-top: 0;">✗ NOT SAFE FOR DIABETES PATIENTS</h3>
                    <p style="font-size: 1.1rem; margin-bottom: 0;">
                        This food is not recommended for people with diabetes. Consider healthier alternatives.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction Confidence", f"{result['confidence']*100:.1f}%")
            
            with col2:
                st.metric("Classification", result['prediction'])
            
            # Probability chart
            st.subheader("Prediction Probability Distribution")
            
            fig = go.Figure(data=[
                go.Bar(
                    x=['Not Safe', 'Safe'],
                    y=[result['probabilities']['Not Safe']*100, 
                       result['probabilities']['Safe']*100],
                    marker_color=['#dc3545', '#28a745'],
                    text=[f"{result['probabilities']['Not Safe']*100:.1f}%",
                          f"{result['probabilities']['Safe']*100:.1f}%"],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Model Confidence Level",
                xaxis_title="Classification",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations
            st.subheader("Nutritional Recommendations")
            
            if result['prediction'] == "Not Safe":
                st.warning("""
                **Recommendations for diabetes patients:**
                - Limit portion size if consuming this food
                - Pair with high-fiber foods to slow glucose absorption
                - Monitor blood sugar levels after consumption
                - Consult with a healthcare provider for personalized advice
                """)
            else:
                st.success("""
                **This food is suitable because:**
                - Low sugar content
                - Moderate carbohydrate levels
                - Appropriate calorie count
                - Can be included in a diabetes-friendly diet
                """)

with tab2:
    st.header("Manual Nutritional Input")
    st.write("Enter nutritional values manually for custom food analysis.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        serving_size = st.number_input("Serving Size (g)", min_value=0.0, value=100.0, step=1.0)
        energy_kcal = st.number_input("Energy (kcal)", min_value=0.0, value=150.0, step=1.0)
        protein_g = st.number_input("Protein (g)", min_value=0.0, value=5.0, step=0.1)
        carbohydrate_g = st.number_input("Carbohydrates (g)", min_value=0.0, value=20.0, step=0.1)
    
    with col2:
        fat_g = st.number_input("Fat (g)", min_value=0.0, value=3.0, step=0.1)
        sugar_g = st.number_input("Sugar (g)", min_value=0.0, value=5.0, step=0.1)
        sodium_mg = st.number_input("Sodium (mg)", min_value=0.0, value=200.0, step=1.0)
        fiber_g = st.number_input("Fiber (g)", min_value=0.0, value=2.0, step=0.1)
    
    if st.button("Analyze Custom Food", type="primary", use_container_width=True):
        manual_input = {
            'serving_size': serving_size,
            'energy_kcal': energy_kcal,
            'protein_g': protein_g,
            'carbohydrate_g': carbohydrate_g,
            'fat_g': fat_g,
            'sugar_g': sugar_g,
            'sodium_mg': sodium_mg,
            'fiber_g': fiber_g
        }
        
        with st.spinner("Analyzing nutritional data..."):
            result = predict_diabetes_safety(manual_input)
        
        st.subheader("Analysis Results")
        
        if result['prediction'] == "Safe":
            st.markdown(f"""
            <div class="result-safe">
                <h3 style="color: #28a745; margin-top: 0;">✓ SAFE FOR DIABETES PATIENTS</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">
                    Based on the nutritional values provided, this food is suitable for diabetes patients.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unsafe">
                <h3 style="color: #dc3545; margin-top: 0;">✗ NOT SAFE FOR DIABETES PATIENTS</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">
                    Based on the nutritional values provided, this food should be avoided by diabetes patients.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result['confidence']*100:.1f}%")
        with col2:
            st.metric("Classification", result['prediction'])

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>Disclaimer:</strong> This system is for educational and informational purposes only. 
    Always consult with healthcare professionals for personalized medical advice.</p>
    <p>Artificial Intelligence Course Project - Food Recommendation Assistant for Diabetes Patients</p>
</div>
""", unsafe_allow_html=True)