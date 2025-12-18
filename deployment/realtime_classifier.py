import cv2
import numpy as np
from pathlib import Path
import joblib
import time

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class RealtimeClassifier:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = None
        self.model_name = None
        self.cnn_extractor = None
        
        self._load_model(model_path)
        self._load_cnn_extractor()
        print(f"Realtime classifier initialized with {self.model_name}")
    
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
    
    def classify_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = self.cnn_extractor.extract_features(frame_rgb).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        
        prediction = self.model.predict(features_scaled)[0]
        proba = self.model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        if confidence < config.CONFIDENCE_THRESHOLD:
            prediction = 6
        
        return prediction, confidence
    
    def draw_overlay(self, frame, class_id, confidence, fps=None):
        class_name = config.CLASSES[class_id]
        color = config.CLASS_COLORS[class_id]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (10, 10), (350, 110), color, 3)
        
        cv2.putText(frame, f"Class: {class_name}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        bar_width = int(300 * confidence)
        cv2.rectangle(frame, (20, 85), (320, 100), (100, 100, 100), -1)
        cv2.rectangle(frame, (20, 85), (20 + bar_width, 100), color, -1)
        
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame
    
    def run(self, camera_id=0):
        print("\nReal-Time Classification")
        print("Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_history = []
        screenshot_count = 0
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            class_id, confidence = self.classify_frame(frame)
            
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            display_frame = self.draw_overlay(frame, class_id, confidence, avg_fps)
            cv2.imshow("Waste Classification", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = config.OUTPUT_DIR / f"screenshot_{screenshot_count}.jpg"
                cv2.imwrite(str(path), display_frame)
                print(f"Screenshot saved: {path}")
                screenshot_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
