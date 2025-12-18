import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class SVMClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.best_params = None
        self.is_fitted = False
        
    def preprocess(self, X, fit=False):
        if fit:
            print(f"  Input features: {X.shape[1]}")
            X_scaled = self.scaler.fit_transform(X)
            print(f"  Features scaled (StandardScaler)")
        else:
            X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def train(self, X_train, y_train, optimize=True, verbose=True):
        if verbose:
            print("\n" + "="*50)
            print("Training SVM Classifier")
            print("="*50)
        
        X_train_scaled = self.preprocess(X_train, fit=True)
        
        if optimize:
            param_grid = config.SVM_PARAM_GRID
            cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
            
            if verbose:
                total = 1
                for v in param_grid.values():
                    total *= len(v)
                print(f"  Grid search: {total} combinations x {config.CV_FOLDS} folds")
            
            svm = SVC(probability=True, random_state=config.RANDOM_STATE, class_weight='balanced', cache_size=2000)
            grid_search = GridSearchCV(svm, param_grid, cv=cv, scoring='f1_macro', n_jobs=1, verbose=1 if verbose else 0)
            grid_search.fit(X_train_scaled, y_train)
            
            self.best_params = grid_search.best_params_
            self.model = grid_search.best_estimator_
            
            if verbose:
                print(f"\n  Best parameters: {self.best_params}")
                print(f"  Best CV Macro F1: {grid_search.best_score_:.4f}")
        else:
            self.model = SVC(C=config.SVM_C, gamma=config.SVM_GAMMA, kernel=config.SVM_KERNEL,
                           probability=True, random_state=config.RANDOM_STATE, class_weight='balanced')
            self.model.fit(X_train_scaled, y_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        return self.model.predict(self.preprocess(X, fit=False))
    
    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        return self.model.predict_proba(self.preprocess(X, fit=False))
    
    def evaluate(self, X_test, y_test, verbose=True):
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
        if verbose:
            print(f"\nSVM Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print("\nClassification Report:")
            unique_labels = sorted(set(y_test) | set(y_pred))
            target_names = [config.CLASSES[i] for i in unique_labels]
            print(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names, zero_division=0))
        
        return {'accuracy': accuracy, 'macro_f1': macro_f1, 'predictions': y_pred, 'probabilities': self.predict_proba(X_test)}
    
    def save(self, path=None):
        if path is None:
            path = config.MODELS_DIR / "svm_model.joblib"
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'best_params': self.best_params}, path)
        print(f"SVM model saved to {path}")
    
    def load(self, path=None):
        if path is None:
            path = config.MODELS_DIR / "svm_model.joblib"
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.best_params = data.get('best_params')
        self.is_fitted = True
        print(f"SVM model loaded from {path}")
        return self
