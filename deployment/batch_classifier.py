import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class BatchClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.model_name = None
        self.cnn_extractor = None
        
        self._load_model(model_path)
        self._load_cnn_extractor()
        print(f"Batch classifier initialized with {self.model_name}")
    
    def _load_model(self, model_path=None):
        if model_path is None:
            info_path = config.MODELS_DIR / "training_info.joblib"
            if info_path.exists():
                info = joblib.load(info_path)
                best_name = info.get('best_model_name', 'svm')
                model_path = config.MODELS_DIR / f"{best_name}_model.joblib"
            else:
                model_path = config.MODELS_DIR / "svm_model.joblib"
        
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_name = Path(model_path).stem
    
    def _load_cnn_extractor(self):
        from features.cnn_feature_extractor import CNNFeatureExtractor
        
        info_path = config.MODELS_DIR / "cnn_extractor_info.joblib"
        if info_path.exists():
            info = joblib.load(info_path)
            model_name = info.get('cnn_model', config.CNN_MODEL)
        else:
            model_name = config.CNN_MODEL
        
        self.cnn_extractor = CNNFeatureExtractor(model_name=model_name)
    
    def classify_image(self, image_path):
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None, None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        features = self.cnn_extractor.extract_features(img).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        if confidence < config.CONFIDENCE_THRESHOLD:
            prediction = 6
        
        return prediction, config.CLASSES[prediction], confidence
    
    def classify_folder(self, folder_path, save_results=True):
        folder_path = Path(folder_path)
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in extensions:
            image_files.extend(list(folder_path.glob(ext)))
            image_files.extend(list(folder_path.glob(ext.upper())))
        
        print(f"Found {len(image_files)} images in {folder_path}")
        
        results = []
        for img_path in tqdm(image_files, desc="Classifying"):
            class_id, class_name, confidence = self.classify_image(img_path)
            
            if class_id is not None:
                results.append({
                    'filename': img_path.name,
                    'filepath': str(img_path),
                    'class_id': class_id,
                    'class_name': class_name,
                    'confidence': confidence
                })
        
        results_df = pd.DataFrame(results)
        
        if save_results and len(results_df) > 0:
            output_path = config.PREDICTIONS_DIR / f"batch_results_{folder_path.name}.csv"
            results_df.to_csv(output_path, index=False)
            print(f"\nResults saved to {output_path}")
        
        return results_df
