import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class UnknownHandler:
    def __init__(self, threshold=None):
        self.threshold = threshold or config.CONFIDENCE_THRESHOLD
        self.rejection_stats = {}
        
    def predict_with_rejection(self, model, X, return_confidence=False):
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        max_probs = np.max(probabilities, axis=1)
        
        final_predictions = predictions.copy()
        rejected_mask = max_probs < self.threshold
        final_predictions[rejected_mask] = 6
        
        self.rejection_stats = {
            'total': len(predictions),
            'rejected': np.sum(rejected_mask),
            'accepted': np.sum(~rejected_mask),
            'rejection_rate': np.mean(rejected_mask)
        }
        
        if return_confidence:
            return final_predictions, max_probs
        return final_predictions
    
    def get_rejection_stats(self):
        return self.rejection_stats
    
    def analyze_confidence(self, probabilities, labels=None):
        max_probs = np.max(probabilities, axis=1)
        
        analysis = {
            'mean_confidence': np.mean(max_probs),
            'std_confidence': np.std(max_probs),
            'min_confidence': np.min(max_probs),
            'max_confidence': np.max(max_probs),
            'below_threshold': np.mean(max_probs < self.threshold),
            'confidence_percentiles': {
                '10%': np.percentile(max_probs, 10),
                '25%': np.percentile(max_probs, 25),
                '50%': np.percentile(max_probs, 50),
                '75%': np.percentile(max_probs, 75),
                '90%': np.percentile(max_probs, 90)
            }
        }
        
        if labels is not None:
            analysis['per_class'] = {}
            for class_id in range(config.NUM_CLASSES):
                mask = labels == class_id
                if np.sum(mask) > 0:
                    class_probs = max_probs[mask]
                    analysis['per_class'][config.CLASSES[class_id]] = {
                        'mean': np.mean(class_probs),
                        'std': np.std(class_probs),
                        'below_threshold': np.mean(class_probs < self.threshold)
                    }
        
        return analysis
    
    def set_threshold(self, threshold):
        self.threshold = threshold
    
    def find_optimal_threshold(self, probabilities, labels, metric='accuracy'):
        max_probs = np.max(probabilities, axis=1)
        predicted_classes = np.argmax(probabilities, axis=1)
        
        thresholds = np.arange(0.3, 0.95, 0.05)
        results = []
        
        for thresh in thresholds:
            final_preds = predicted_classes.copy()
            rejected_mask = max_probs < thresh
            final_preds[rejected_mask] = 6
            
            accepted_mask = ~rejected_mask
            if np.sum(accepted_mask) > 0:
                accuracy = np.mean(final_preds[accepted_mask] == labels[accepted_mask])
            else:
                accuracy = 0
            
            rejection_rate = np.mean(rejected_mask)
            
            results.append({
                'threshold': thresh,
                'accuracy': accuracy,
                'rejection_rate': rejection_rate,
                'accepted_count': np.sum(accepted_mask)
            })
        
        if metric == 'accuracy':
            optimal = max(results, key=lambda x: x['accuracy'] if x['rejection_rate'] < 0.3 else 0)
        else:
            optimal = max(results, key=lambda x: x['accuracy'] * (1 - x['rejection_rate']))
        
        return optimal, results
