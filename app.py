import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Food Recommendation System for Diabetes",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d2fa30;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-excellent {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000;
    }
    .result-good {
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000;
    }
    .result-moderate {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000;
    }
    .result-caution {
        background-color: #ffe5d0;
        border-left: 5px solid #fd7e14;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000;
    }
    .result-avoid {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 5px;
        margin: 1rem 0;
        color: #000;
    }
            
.stButton > button[kind="primary"] {
    background-color: #130F27 !important;  
    color: #d2fa30 !important; 
    border: 1px solid #3C3D37 !important; 
    font-weight: bold !important;
}
    
.stButton > button[kind="primary"]:hover {
    background-color: #1a1535 !important;
    color: #d2fa30 !important;
    border: 1px solid #3C3D37 !important;
}

.stButton > button[kind="primary"]:active {
    background-color: #d2fa30 !important;
    color: #130F27 !important;
    border: 1px solid #130F27 !important;
}

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #393E46 !important;
        font-weight: bold !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #393E46 !important;
    }
    
    .stTabs [aria-selected="true"] {
        color: #d2fa30 !important; 
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #d2fa30 !important; 
    }
    
    .stTabs [data-baseweb="tab-border"] {
        background-color: #393E46 !important; 
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load('best_diabetes_food_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Model files not found. Please run the training notebook first.")
        st.stop()

model, scaler, feature_names = load_model_and_scaler()

# Category mapping
category_labels = {
    4: 'Excellent',
    3: 'Good', 
    2: 'Moderate',
    1: 'Caution',
    0: 'Avoid'
}

category_colors = {
    'Excellent': '#28a745',
    'Good': '#17a2b8',
    'Moderate': '#ffc107',
    'Caution': '#fd7e14',
    'Avoid': '#dc3545'
}

category_icons = {
    'Excellent': '✓',
    'Good': '✓',
    'Moderate': '⚠',
    'Caution': '⚠',
    'Avoid': '✗'
}

recommendations = {
    'Excellent': "✓ Sangat aman! Boleh dikonsumsi secara regular.",
    'Good': "✓ Aman untuk dikonsumsi. Pilihan yang bagus.",
    'Moderate': "⚠ Boleh sesekali, perhatikan ukuran porsi.",
    'Caution': "⚠ Batasi konsumsi, pilih alternatif lebih sehat.",
    'Avoid': "✗ Sebaiknya dihindari atau konsumsi sangat jarang."
}

# Prediction function
def predict_diabetes_safety(input_dict):
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[feature_names]
    input_scaled = scaler.transform(input_df)
    
    prediction_label = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]
    
    all_probs = {}
    for idx in range(len(probabilities)):
        all_probs[category_labels[idx]] = float(probabilities[idx] * 100)
    
    return {
        'category': category_labels[prediction_label],
        'category_code': int(prediction_label),
        'confidence': float(probabilities[prediction_label] * 100),
        'all_probabilities': all_probs,
        'recommendation': recommendations[category_labels[prediction_label]]
    }

# Header
st.markdown('<p class="main-header">Food Recommendation System for Diabetes Patients</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">5-Level Nutritional Classification System</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About This System")
    st.write("""
    This application uses machine learning to classify foods into 5 safety levels for diabetes patients.
    
    **5 Categories:**
    - Excellent: Sangat aman
    - Good: Aman
    - Moderate: Hati-hati
    - Caution: Batasi
    - Avoid: Hindari
    
    **Model:** Random Forest Classifier (5 classes)
    """)
    
    st.divider()
    
    st.subheader("How to Use")
    st.write("""
    1. Pilih makanan dari dropdown
    2. Review informasi nutrisi
    3. Klik 'Analyze Food Safety'
    4. Lihat hasil kategori & rekomendasi
    """)

# Main content
tab1, tab2 = st.tabs(["Food Analysis", "Manual Input"])

with tab1:
    st.header("Select Food for Analysis")
    
    @st.cache_data
    def load_food_data():
        try:
            df = pd.read_csv('nilai-gizi.csv')
            for col in feature_names:
                if col in df.columns and df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except FileNotFoundError:
            st.error("Dataset not found.")
            return None
    
    df = load_food_data()
    
    if df is not None:
        food_names = df['name'].dropna().unique().tolist()
        selected_food = st.selectbox("Choose a food item:", options=food_names, index=0)
        
        food_data = df[df['name'] == selected_food].iloc[0]
        
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
        
        if st.button("Analyze Food Safety", type="primary", use_container_width=True):
            input_data = {feature: food_data[feature] for feature in feature_names}
            
            with st.spinner("Analyzing..."):
                result = predict_diabetes_safety(input_data)
            
            st.subheader("Analysis Results")
            
            category = result['category']
            icon = category_icons[category]
            color = category_colors[category]
            
            css_class = f"result-{category.lower()}"
            
            st.markdown(f"""
            <div class="{css_class}">
                <h3 style="color: {color}; margin-top: 0;">{icon} {category.upper()}</h3>
                <p style="font-size: 1.1rem; margin-bottom: 0;">
                    {result['recommendation']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Prediction Confidence", f"{result['confidence']:.1f}%")
            
            with col2:
                st.metric("Category", category)
            
            st.subheader("All Category Probabilities")
            
            categories = ['Excellent', 'Good', 'Moderate', 'Caution', 'Avoid']
            probs = [result['all_probabilities'][cat] for cat in categories]
            colors_list = [category_colors[cat] for cat in categories]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=probs,
                    marker_color=colors_list,
                    text=[f"{p:.1f}%" for p in probs],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Probability Distribution Across All Categories",
                xaxis_title="Category",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Detailed Probabilities")
            for cat in categories:
                prob = result['all_probabilities'][cat]
                st.write(f"**{cat}**: {prob:.2f}%")

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
        
        with st.spinner("Analyzing..."):
            result = predict_diabetes_safety(manual_input)
        
        st.subheader("Analysis Results")
        
        category = result['category']
        icon = category_icons[category]
        color = category_colors[category]
        css_class = f"result-{category.lower()}"
        
        st.markdown(f"""
        <div class="{css_class}">
            <h3 style="color: {color}; margin-top: 0;">{icon} {category.upper()}</h3>
            <p style="font-size: 1.1rem; margin-bottom: 0;">
                {result['recommendation']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{result['confidence']:.1f}%")
        with col2:
            st.metric("Category", category)
        
        st.subheader("All Probabilities")
        categories = ['Excellent', 'Good', 'Moderate', 'Caution', 'Avoid']
        probs = [result['all_probabilities'][cat] for cat in categories]
        colors_list = [category_colors[cat] for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=probs,
                marker_color=colors_list,
                text=[f"{p:.1f}%" for p in probs],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Probability Distribution",
            xaxis_title="Category",
            yaxis_title="Probability (%)",
            yaxis_range=[0, 100],
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p><strong>FAKULTAS VOKASI UNIVERSITAS BRAWIJAYA</strong></p>
    <p> Mata Kuliah Kecerdasan Buatan - 2025</p>
    <p> Food Recommendation for Diabetes Patients (5-Level Classification)</p>
</div>
""", unsafe_allow_html=True)