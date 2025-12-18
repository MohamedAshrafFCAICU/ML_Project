import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from deployment.batch_classifier import BatchClassifier
from config.settings import config


def main():
    parser = argparse.ArgumentParser(description='Waste Classification')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--folder', type=str, help='Folder containing images')
    parser.add_argument('--output', type=str, help='Output CSV file')
    parser.add_argument('--model', type=str, default=None, choices=['svm', 'knn'])
    
    args = parser.parse_args()
    
    model_path = None
    if args.model:
        model_path = config.MODELS_DIR / f"{args.model}_model.joblib"
    
    classifier = BatchClassifier(model_path=model_path)
    
    if args.image:
        class_id, class_name, confidence = classifier.classify_image(args.image)
        
        if class_id is None:
            print(f"Error: Could not read image {args.image}")
            return
        
        print(f"\nImage: {args.image}")
        print(f"Class: {class_name} (ID: {class_id})")
        print(f"Confidence: {confidence:.2%}")
    
    elif args.folder:
        results = classifier.classify_folder(args.folder, save_results=True)
        print(f"\nProcessed {len(results)} images")
        
        if args.output:
            results.to_csv(args.output, index=False)
            print(f"Results saved to {args.output}")
    
    else:
        print("Error: Specify --image or --folder")


if __name__ == "__main__":
    main()
