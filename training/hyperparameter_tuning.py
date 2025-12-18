import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class HyperparameterTuner:
    def __init__(self):
        self.best_params = {}
        self.cv_results = {}
        
    def tune_svm(self, X, y, param_grid=None, cv=None, verbose=True):
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100, 500, 1000],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'class_weight': ['balanced', None]
            }
        
        cv = cv or StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        if verbose:
            print("\nTuning SVM hyperparameters...")
        
        svm = SVC(probability=True, random_state=config.RANDOM_STATE)
        grid_search = RandomizedSearchCV(svm, param_grid, n_iter=50, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2 if verbose else 0, random_state=config.RANDOM_STATE)
        grid_search.fit(X, y)
        
        self.best_params['svm'] = grid_search.best_params_
        
        if verbose:
            print(f"\nBest SVM parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def tune_knn(self, X, y, param_grid=None, cv=None, verbose=True):
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'cosine']
            }
        
        cv = cv or StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        
        if verbose:
            print("\nTuning k-NN hyperparameters...")
        
        knn = KNeighborsClassifier()
        grid_search = RandomizedSearchCV(knn, param_grid, n_iter=50, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2 if verbose else 0, random_state=config.RANDOM_STATE)
        grid_search.fit(X, y)
        
        self.best_params['knn'] = grid_search.best_params_
        
        if verbose:
            print(f"\nBest k-NN parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_params_, grid_search.best_score_
    
    def save_results(self, path=None):
        if path is None:
            path = config.MODELS_DIR / "tuning_results.joblib"
        joblib.dump({'best_params': self.best_params}, path)
        print(f"Tuning results saved to {path}")
