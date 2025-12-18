import cv2
import numpy as np
import time
import joblib
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from features.cnn_feature_extractor import CNNFeatureExtractor


class WebcamClassifier:
    def __init__(self):
        self.cnn_extractor = None
        self.svm_model = None
        self.svm_scaler = None
        self.knn_model = None
        self.knn_scaler = None
        self.current_model = 'svm'
        self._load_models()
    
    def _load_models(self):
        print("Loading models...")
        self.cnn_extractor = CNNFeatureExtractor()
        
        svm_path = config.MODELS_DIR / "svm_model.joblib"
        if svm_path.exists():
            data = joblib.load(svm_path)
            self.svm_model = data['model']
            self.svm_scaler = data['scaler']
            print("  SVM model loaded")
        
        knn_path = config.MODELS_DIR / "knn_model.joblib"
        if knn_path.exists():
            data = joblib.load(knn_path)
            self.knn_model = data['model']
            self.knn_scaler = data['scaler']
            print("  KNN model loaded")
        
        print("Models loaded!")
    
    def classify(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        features = self.cnn_extractor.extract_features(rgb).reshape(1, -1)
        
        if self.current_model == 'svm' and self.svm_model:
            model, scaler = self.svm_model, self.svm_scaler
        elif self.knn_model:
            model, scaler = self.knn_model, self.knn_scaler
        else:
            return None, None, None
        
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        
        if confidence <= 0.35:
            prediction = 6
            class_name = 'unknown'
        else:
            class_name = config.CLASSES[prediction]
        
        return prediction, class_name, confidence
    
    def draw_ui(self, frame, class_name, confidence, fps, paused):
        h, w = frame.shape[:2]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        color = config.CLASS_COLORS.get(config.get_class_id(class_name) if class_name else 6, (255, 255, 255))
        color = (color[2], color[1], color[0])
        
        cv2.rectangle(frame, (10, 10), (350, 120), color, 3)
        
        if class_name:
            cv2.putText(frame, class_name.upper(), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        if confidence:
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            bar_width = int(300 * confidence)
            cv2.rectangle(frame, (20, 90), (320, 105), (50, 50, 50), -1)
            cv2.rectangle(frame, (20, 90), (20 + bar_width, 105), color, -1)
        
        cv2.putText(frame, f"Model: {self.current_model.upper()}", (w - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if paused:
            cv2.putText(frame, "PAUSED", (w//2 - 60, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return frame
    
    def run(self, camera_id=0):
        print("\n" + "="*50)
        print("WEBCAM WASTE CLASSIFICATION")
        print("Q: Quit | S: Screenshot | M: Switch Model | Space: Pause")
        print("="*50 + "\n")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Cannot open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps_history = []
        screenshot_count = 0
        paused = False
        last_result = (None, None, None)
        
        while True:
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            if not paused:
                class_id, class_name, confidence = self.classify(frame)
                last_result = (class_id, class_name, confidence)
            else:
                class_id, class_name, confidence = last_result
            
            elapsed = time.time() - start_time
            fps = 1.0 / elapsed if elapsed > 0 else 0
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)
            
            display = self.draw_ui(frame, class_name, confidence, avg_fps, paused)
            cv2.imshow("Waste Classification", display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                path = config.OUTPUT_DIR / f"webcam_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(str(path), display)
                print(f"Screenshot saved: {path}")
                screenshot_count += 1
            elif key == ord('m'):
                if self.current_model == 'svm' and self.knn_model:
                    self.current_model = 'knn'
                elif self.svm_model:
                    self.current_model = 'svm'
                print(f"Switched to {self.current_model.upper()}")
            elif key == ord(' '):
                paused = not paused
        
        cap.release()
        cv2.destroyAllWindows()


def main():
    classifier = WebcamClassifier()
    classifier.run()


if __name__ == "__main__":
    main()
