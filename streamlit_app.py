import streamlit as st
import cv2
import numpy as np
import joblib
from pathlib import Path
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from features.cnn_feature_extractor import CNNFeatureExtractor

st.set_page_config(
    page_title="Waste Classification",
    page_icon="♻️",
    layout="wide"
)

CLASS_COLORS = {
    'cardboard': '#8B4513',
    'glass': '#00FF00',
    'metal': '#C0C0C0',
    'paper': '#FFD700',
    'plastic': '#1E90FF',
    'trash': '#808080',
    'unknown': '#FF00FF'
}

@st.cache_resource
def load_models():
    cnn_extractor = CNNFeatureExtractor()
    
    svm_model, svm_scaler = None, None
    knn_model, knn_scaler = None, None
    
    svm_path = config.MODELS_DIR / "svm_model.joblib"
    if svm_path.exists():
        data = joblib.load(svm_path)
        svm_model = data['model']
        svm_scaler = data['scaler']
    
    knn_path = config.MODELS_DIR / "knn_model.joblib"
    if knn_path.exists():
        data = joblib.load(knn_path)
        knn_model = data['model']
        knn_scaler = data['scaler']
    
    return cnn_extractor, svm_model, svm_scaler, knn_model, knn_scaler

def classify_image(image, model_type, cnn_extractor, svm_model, svm_scaler, knn_model, knn_scaler):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    features = cnn_extractor.extract_features(image).reshape(1, -1)
    
    if model_type == 'SVM' and svm_model:
        model, scaler = svm_model, svm_scaler
    elif knn_model:
        model, scaler = knn_model, knn_scaler
    else:
        return None, None, None
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    proba = model.predict_proba(features_scaled)[0]
    confidence = float(np.max(proba))
    
    if confidence <= 0.35:
        class_name = 'unknown'
    else:
        class_name = config.CLASSES[prediction]
    
    all_probs = {config.CLASSES[i]: float(p) for i, p in enumerate(proba)}
    
    return class_name, confidence, all_probs

def main():
    st.title("♻️ Waste Classification System")
    st.write("AI-powered waste classification using EfficientNet + SVM/KNN")
    
    with st.spinner("Loading AI models..."):
        cnn_extractor, svm_model, svm_scaler, knn_model, knn_scaler = load_models()
    
    with st.sidebar:
        st.header("Settings")
        
        available_models = []
        if svm_model:
            available_models.append("SVM")
        if knn_model:
            available_models.append("KNN")
        
        if available_models:
            model_type = st.selectbox("Select Model", available_models)
        else:
            st.error("No models found!")
            return
        
        st.markdown("---")
        st.header("Classes")
        for class_name, color in CLASS_COLORS.items():
            if class_name != 'unknown':
                st.markdown(f"**{class_name.upper()}**")
    
    tab1, tab2 = st.tabs(["📷 Camera", "🖼️ Upload Image"])
    
    with tab1:
        st.subheader("Camera Classification")
        camera_image = st.camera_input("Take a photo")
        
        if camera_image:
            image = Image.open(camera_image)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Captured Image", use_container_width=True)
            
            with col2:
                with st.spinner("Classifying..."):
                    class_name, confidence, all_probs = classify_image(
                        image, model_type, cnn_extractor, svm_model, svm_scaler, knn_model, knn_scaler
                    )
                
                if class_name:
                    color = CLASS_COLORS.get(class_name, '#00ff88')
                    st.markdown(f"### Result: **{class_name.upper()}**")
                    st.markdown(f"Confidence: **{confidence:.1%}**")
                    
                    st.subheader("All Probabilities")
                    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                        st.progress(prob, text=f"{cls}: {prob:.1%}")
    
    with tab2:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            
            with col2:
                if st.button("🔍 Classify", use_container_width=True):
                    with st.spinner("Classifying..."):
                        class_name, confidence, all_probs = classify_image(
                            image, model_type, cnn_extractor, svm_model, svm_scaler, knn_model, knn_scaler
                        )
                    
                    if class_name:
                        st.markdown(f"### Result: **{class_name.upper()}**")
                        st.markdown(f"Confidence: **{confidence:.1%}**")
                        
                        st.subheader("All Probabilities")
                        for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                            st.progress(prob, text=f"{cls}: {prob:.1%}")

if __name__ == "__main__":
    main()
