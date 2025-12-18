import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config
from models.svm_classifier import SVMClassifier
from models.knn_classifier import KNNClassifier


class ModelTrainer:
    def __init__(self):
        self.svm_model = None
        self.knn_model = None
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def split_data(self, features, labels, test_size=None, val_size=None):
        test_size = test_size or config.TEST_SIZE
        val_size = val_size or config.VAL_SIZE
        
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            features, labels, test_size=test_size, stratify=labels, random_state=config.RANDOM_STATE
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=config.RANDOM_STATE
        )
        
        print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_svm(self, X_train, y_train, X_val, y_val, optimize=True):
        self.svm_model = SVMClassifier()
        self.svm_model.train(X_train, y_train, optimize=optimize)
        val_results = self.svm_model.evaluate(X_val, y_val, verbose=True)
        self.results['svm'] = val_results
        return self.svm_model, val_results
    
    def train_knn(self, X_train, y_train, X_val, y_val, optimize=True):
        self.knn_model = KNNClassifier()
        self.knn_model.train(X_train, y_train, optimize=optimize)
        val_results = self.knn_model.evaluate(X_val, y_val, verbose=True)
        self.results['knn'] = val_results
        return self.knn_model, val_results
    
    def train_all(self, X_train, y_train, X_val, y_val, optimize=True):
        print("\n" + "="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.train_svm(X_train, y_train, X_val, y_val, optimize)
        self.train_knn(X_train, y_train, X_val, y_val, optimize)
        self._find_best_model()
        
        return self.results
    
    def _find_best_model(self):
        accuracies = {
            'svm': self.results.get('svm', {}).get('accuracy', 0),
            'knn': self.results.get('knn', {}).get('accuracy', 0)
        }
        
        self.best_model_name = max(accuracies, key=accuracies.get)
        self.best_model = self.svm_model if self.best_model_name == 'svm' else self.knn_model
        
        print(f"\nBest Model: {self.best_model_name.upper()}")
    
    def evaluate_on_test(self, X_test, y_test):
        print("\n" + "="*60)
        print("FINAL TEST EVALUATION")
        print("="*60)
        return self.best_model.evaluate(X_test, y_test, verbose=True)
    
    def save_models(self):
        print("\nSaving models...")
        if self.svm_model:
            self.svm_model.save()
        if self.knn_model:
            self.knn_model.save()
        
        joblib.dump({
            'best_model_name': self.best_model_name,
            'results': {k: {'accuracy': v.get('accuracy', 0)} for k, v in self.results.items()}
        }, config.MODELS_DIR / "training_info.joblib")
        
        print("All models saved!")
