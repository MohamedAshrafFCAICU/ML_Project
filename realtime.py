import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from deployment.realtime_classifier import RealtimeClassifier

if __name__ == "__main__":
    print("="*60)
    print("   REAL-TIME CLASSIFICATION")
    print("="*60)
    
    classifier = RealtimeClassifier()
    classifier.run()
