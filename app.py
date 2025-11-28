import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Food Recommendation for Diabetes",
    page_icon="🍎",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Load model ML dan preprocessing tools"""
    try:
        model = joblib.load('best_diabetes_food_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Cek apakah ada target encoder
        if os.path.exists('target_encoder.pkl'):
            encoder = joblib.load('target_encoder.pkl')
        else:
            encoder = None
        
        return model, scaler, encoder
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

@st.cache_data
def load_food_database():
    """Load database makanan dari CSV"""
    try:
        df = pd.read_csv('pred_food.csv')
        return df
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return None

# Load model dan data
model, scaler, encoder = load_model()
food_df = load_food_database()

# Kolom fitur yang digunakan (harus sama dengan saat training)
FEATURE_COLS = [
    'Glycemic Index', 'Calories', 'Carbohydrates', 'Protein', 'Fat',
    'Sodium Content', 'Potassium Content', 'Magnesium Content',
    'Calcium Content', 'Fiber Content'
]


def predict_food(food_data):
    """
    Prediksi kesesuaian makanan untuk penderita diabetes
    
    Parameters:
    -----------
    food_data : pandas Series
        Data makanan dengan fitur nutrisi
    
    Returns:
    --------
    dict : hasil prediksi
    """
    # Ambil fitur yang diperlukan
    features = food_data[FEATURE_COLS].values.reshape(1, -1)
    
    # Scaling
    features_scaled = scaler.transform(features)
    
    # Prediksi
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Decode label jika ada encoder
    if encoder is not None:
        label = encoder.inverse_transform([prediction])[0]
    else:
        label = "Yes" if prediction == 1 else "No"
    
    return {
        'prediction': int(prediction),
        'label': label,
        'probability': probability,
        'confidence': float(max(probability))
    }

def get_recommendation_message(result, food_name):
    """Generate pesan rekomendasi berdasarkan hasil prediksi"""
    is_suitable = (result['label'] == 'Yes' or result['prediction'] == 1)
    confidence = result['confidence']
    
    if is_suitable:
        if confidence >= 0.8:
            return f"**AMAN DIKONSUMSI**\n\n'{food_name}' sangat cocok untuk penderita diabetes dengan tingkat keyakinan {confidence:.1%}. Makanan ini memiliki kandungan nutrisi yang sesuai."
        else:
            return f"**RELATIF AMAN**\n\n'{food_name}' cukup cocok untuk penderita diabetes (keyakinan {confidence:.1%}), namun tetap perhatikan porsi konsumsi."
    else:
        if confidence >= 0.8:
            return f"**TIDAK DIREKOMENDASIKAN**\n\n'{food_name}' tidak cocok untuk penderita diabetes dengan tingkat keyakinan {confidence:.1%}. Sebaiknya hindari atau batasi konsumsi makanan ini."
        else:
            return f"**KURANG DIREKOMENDASIKAN**\n\n'{food_name}' kurang cocok untuk penderita diabetes (keyakinan {confidence:.1%}). Konsumsi dengan hati-hati dan dalam porsi kecil."



# Header
st.title("Food Recommendation Assistant")
st.subheader("Sistem Rekomendasi Makanan untuk Penderita Diabetes")
st.markdown("---")

# Cek apakah model dan data berhasil di-load
if model is None or food_df is None:
    st.error("Gagal memuat model atau database. Pastikan semua file tersedia!")
    st.stop()

# Informasi dataset
st.sidebar.header("Informasi Database")
st.sidebar.info(f"Total makanan dalam database: **{len(food_df)}**")
st.sidebar.markdown("---")

# Pilihan mode
st.sidebar.header("Mode Aplikasi")
mode = st.sidebar.radio(
    "Pilih mode:",
    ["Cari Makanan", "Input Manual"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Tentang Aplikasi")
st.sidebar.info(
    "Aplikasi ini menggunakan Machine Learning untuk "
    "memprediksi apakah suatu makanan cocok atau tidak "
    "untuk penderita diabetes berdasarkan kandungan nutrisinya."
)


if mode == "Cari Makanan":
    st.header("Cari Makanan dari Database")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Dropdown untuk memilih makanan
        food_list = sorted(food_df['Food Name'].unique().tolist())
        selected_food = st.selectbox(
            "Pilih nama makanan:",
            options=food_list,
            help="Pilih makanan dari dropdown atau ketik untuk mencari"
        )
    
    with col2:
        st.markdown("##")  # Spacing
        predict_button = st.button("Cek Kesesuaian", type="primary", use_container_width=True)
    
    if predict_button and selected_food:
        # Ambil data makanan yang dipilih
        food_data = food_df[food_df['Food Name'] == selected_food].iloc[0]
        
        # Tampilkan info makanan
        st.markdown("---")
        st.subheader(f"Informasi Nutrisi: {selected_food}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Glycemic Index", f"{food_data['Glycemic Index']:.0f}")
            st.metric("Kalori", f"{food_data['Calories']:.0f} kcal")
            st.metric("Karbohidrat", f"{food_data['Carbohydrates']:.1f} g")
            st.metric("Protein", f"{food_data['Protein']:.1f} g")
        
        with col2:
            st.metric("Lemak", f"{food_data['Fat']:.1f} g")
            st.metric("Sodium", f"{food_data['Sodium Content']:.0f} mg")
            st.metric("Potassium", f"{food_data['Potassium Content']:.0f} mg")
            st.metric("Magnesium", f"{food_data['Magnesium Content']:.0f} mg")
        
        with col3:
            st.metric("Kalsium", f"{food_data['Calcium Content']:.0f} mg")
            st.metric("Serat", f"{food_data['Fiber Content']:.1f} g")
        
        # Prediksi
        with st.spinner("Menganalisis makanan..."):
            result = predict_food(food_data)
        
        st.markdown("---")
        st.subheader("Hasil Analisis")
        
        # Tampilkan rekomendasi
        message = get_recommendation_message(result, selected_food)
        
        if result['prediction'] == 1:
            st.success(message)
        else:
            st.warning(message)
        
        # Progress bar confidence
        st.markdown("### Tingkat Keyakinan Model")
        st.progress(result['confidence'])
        st.caption(f"Model yakin {result['confidence']:.1%} dengan prediksi ini")
        
        # Detail probabilitas
        with st.expander("Lihat Detail Probabilitas"):
            prob_df = pd.DataFrame({
                'Kategori': ['Tidak Cocok', 'Cocok'],
                'Probabilitas': [f"{result['probability'][0]:.2%}", f"{result['probability'][1]:.2%}"]
            })
            st.table(prob_df)


else:
    st.header("Input Data Nutrisi Manual")
    st.info("Mode ini untuk makanan yang tidak ada dalam database. Masukkan data nutrisi secara manual.")
    
    with st.form("manual_input_form"):
        food_name_manual = st.text_input(
            "Nama Makanan:",
            placeholder="Contoh: Nasi Goreng Spesial"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            gi = st.number_input("Glycemic Index", 0, 100, 50, help="0-55: Rendah, 56-69: Sedang, 70+: Tinggi")
            calories = st.number_input("Kalori (kcal)", 0, 1000, 150)
            carbs = st.number_input("Karbohidrat (g)", 0.0, 200.0, 30.0, step=0.1)
            protein = st.number_input("Protein (g)", 0.0, 100.0, 10.0, step=0.1)
            fat = st.number_input("Lemak (g)", 0.0, 100.0, 5.0, step=0.1)
        
        with col2:
            sodium = st.number_input("Sodium (mg)", 0, 2000, 100)
            potassium = st.number_input("Potassium (mg)", 0, 2000, 250)
            magnesium = st.number_input("Magnesium (mg)", 0, 500, 50)
            calcium = st.number_input("Kalsium (mg)", 0, 1000, 100)
            fiber = st.number_input("Serat (g)", 0.0, 50.0, 8.0, step=0.1)
        
        submit_button = st.form_submit_button("Analisis Makanan", type="primary", use_container_width=True)
    
    if submit_button:
        if not food_name_manual:
            st.warning("Mohon masukkan nama makanan terlebih dahulu!")
        else:
            # Buat DataFrame untuk input manual
            manual_data = pd.Series({
                'Food Name': food_name_manual,
                'Glycemic Index': gi,
                'Calories': calories,
                'Carbohydrates': carbs,
                'Protein': protein,
                'Fat': fat,
                'Sodium Content': sodium,
                'Potassium Content': potassium,
                'Magnesium Content': magnesium,
                'Calcium Content': calcium,
                'Fiber Content': fiber
            })
            
            # Prediksi
            with st.spinner("Menganalisis makanan..."):
                result = predict_food(manual_data)
            
            st.markdown("---")
            st.subheader("Hasil Analisis")
            
            # Tampilkan rekomendasi
            message = get_recommendation_message(result, food_name_manual)
            
            if result['prediction'] == 1:
                st.success(message)
            else:
                st.warning(message)
            
            # Progress bar confidence
            st.markdown("### Tingkat Keyakinan Model")
            st.progress(result['confidence'])
            st.caption(f"Model yakin {result['confidence']:.1%} dengan prediksi ini")
            
            # Detail probabilitas
            with st.expander("Lihat Detail Probabilitas"):
                prob_df = pd.DataFrame({
                    'Kategori': ['Tidak Cocok', 'Cocok'],
                    'Probabilitas': [f"{result['probability'][0]:.2%}", f"{result['probability'][1]:.2%}"]
                })
                st.table(prob_df)


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>MATA KULIAH KECERDASAN BUATAN | Dataset: Diabetes Food Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)