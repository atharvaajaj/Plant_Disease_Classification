import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

# Load and preprocess the image
def model_predict(image_path):
    model = tf.keras.models.load_model('plant_disease_cnn_model.keras')
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)  # Reshape for model input

    predictions = model.predict(img)
    confidence = np.max(predictions) * 100  # Get highest probability
    predicted_class = np.argmax(predictions)

    return predicted_class, confidence  # Return both values

# Custom Styling
st.markdown("""
    <style>
        .main {background-color: #f0f2f6;}
        h1 {color: #2E8B57; text-align: center;}
        .stButton>button {background-color: #4CAF50; color: white; font-size: 16px; border-radius: 10px;}
        .stButton>button:hover {background-color: #45a049;}
        .stSidebar {background-color: #ddeedd;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.title('🌿 Plant Disease Prediction System')
st.sidebar.subheader("ℹ️ About This App")
st.sidebar.write("""
- This app uses a **Deep Learning CNN Model** to detect plant diseases from images.  
- Model trained on **New Plant Diseases Dataset (Kaggle)**.  
- **Accuracy:** ~91.48%  
- **Supports multiple plant types & diseases.**
""")

app_mode = st.sidebar.radio('Select Page', ['🏠 Home', '🔍 Disease Recognition'])

img = Image.open('Disease.png')
st.image(img)

if app_mode == '🏠 Home':
    st.markdown("<h1>🌱 Plant Disease Detection for Sustainable Agriculture</h1>", unsafe_allow_html=True)

elif app_mode == '🔍 Disease Recognition':
    st.header("📸 Upload an Image for Prediction")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, 'wb') as f:
            f.write(test_image.getbuffer())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(test_image, width=250, caption="Uploaded Image")

        with col2:
            if st.button("🚀 Predict Disease"):
                st.write("🔍 **Analyzing...**")
                result_index, confidence = model_predict(save_path)
                class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                              'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                              'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                              'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                              'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                              'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                              'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                              'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                              'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                              'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                              'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                              'Tomato___healthy']
                
                st.success(f"🌿 **Detected Disease:** {class_name[result_index]}")
                st.info(f"📊 **Confidence Score:** {confidence:.2f}%")