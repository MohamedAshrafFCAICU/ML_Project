import os
import cv2
import numpy as np
import base64
import joblib
from pathlib import Path
from flask import Flask, render_template, request, jsonify

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from features.cnn_feature_extractor import CNNFeatureExtractor

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

classifier = None


class WebClassifier:
    def __init__(self):
        self.cnn_extractor = None
        self.svm_model = None
        self.svm_scaler = None
        self.knn_model = None
        self.knn_scaler = None
        self._load_models()
    
    def _load_models(self):
        print("Loading models...")
        self.cnn_extractor = CNNFeatureExtractor()
        
        svm_path = config.MODELS_DIR / "svm_model.joblib"
        if svm_path.exists():
            data = joblib.load(svm_path)
            self.svm_model = data['model']
            self.svm_scaler = data['scaler']
            print("  SVM loaded")
        
        knn_path = config.MODELS_DIR / "knn_model.joblib"
        if knn_path.exists():
            data = joblib.load(knn_path)
            self.knn_model = data['model']
            self.knn_scaler = data['scaler']
            print("  KNN loaded")
        
        print("Models ready!")
    
    def classify(self, image, model_type='svm'):
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        
        features = self.cnn_extractor.extract_features(rgb).reshape(1, -1)
        
        if model_type == 'svm' and self.svm_model:
            model, scaler = self.svm_model, self.svm_scaler
        elif self.knn_model:
            model, scaler = self.knn_model, self.knn_scaler
        else:
            return {'error': 'No model loaded'}
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba))
        
        if confidence <= 0.35:
            prediction = 6
            class_name = 'unknown'
        else:
            class_name = config.CLASSES[prediction]
        
        all_probs = {config.CLASSES[i]: float(p) for i, p in enumerate(proba)}
        
        return {
            'class_id': int(prediction),
            'class_name': class_name,
            'confidence': confidence,
            'probabilities': all_probs
        }


def init_classifier():
    global classifier
    if classifier is None:
        classifier = WebClassifier()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/classify', methods=['POST'])
def classify_image():
    init_classifier()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    model_type = request.form.get('model', 'svm')
    
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    result = classifier.classify(img, model_type)
    return jsonify(result)


@app.route('/api/classify_base64', methods=['POST'])
def classify_base64():
    init_classifier()
    
    data = request.json
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    model_type = data.get('model', 'svm')
    
    img_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid image'}), 400
    
    result = classifier.classify(img, model_type)
    return jsonify(result)


@app.route('/api/models')
def get_models():
    init_classifier()
    models = []
    if classifier.svm_model:
        models.append('svm')
    if classifier.knn_model:
        models.append('knn')
    return jsonify({'models': models})


@app.route('/api/classes')
def get_classes():
    classes = []
    for i, name in config.CLASSES.items():
        color = config.CLASS_COLORS.get(i, (128, 128, 128))
        classes.append({'id': i, 'name': name, 'color': f'rgb({color[0]}, {color[1]}, {color[2]})'})
    return jsonify({'classes': classes})


if __name__ == '__main__':
    print("\n" + "="*50)
    print("WASTE CLASSIFICATION WEB APP")
    print("="*50 + "\n")
    
    init_classifier()
    
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
