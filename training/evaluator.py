import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class ModelEvaluator:
    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else config.FIGURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def comprehensive_evaluation(self, y_true, y_pred, y_proba=None):
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        return results
    
    def print_report(self, y_true, y_pred, model_name="Model"):
        print(f"\n{'='*60}")
        print(f"Classification Report - {model_name}")
        print("="*60)
        
        target_names = [config.CLASSES[i] for i in range(config.NUM_CLASSES)]
        print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))
        
        results = self.comprehensive_evaluation(y_true, y_pred)
        print(f"\nAccuracy: {results['accuracy']:.4f}")
        return results
    
    def plot_confusion_matrix(self, y_true, y_pred, model_name="Model", save=True):
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        target_names = [config.CLASSES[i] for i in range(config.NUM_CLASSES)]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.close()
