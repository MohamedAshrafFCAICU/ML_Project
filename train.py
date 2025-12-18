import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import train_models

if __name__ == "__main__":
    print("="*60)
    print("   TRAINING")
    print("="*60)
    
    results = train_models()
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best Model: {results['best_model'].upper()}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print("="*60)
