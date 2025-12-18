import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config.settings import config
from utils.logger import logger


def train_models():
    from data.data_loader import DataLoader
    from data.data_cleaner import DataCleaner
    from data.data_augmentor import DataAugmentor
    from data.data_visualizer import DataVisualizer
    from features.cnn_feature_extractor import CNNFeatureExtractor
    from models.svm_classifier import SVMClassifier
    from models.knn_classifier import KNNClassifier
    from training.evaluator import ModelEvaluator
    from sklearn.model_selection import train_test_split
    import joblib
    
    logger.info("="*60)
    logger.info("TRAINING PIPELINE")
    logger.info("="*60)
    
    logger.info("\n[STEP 1] Loading dataset...")
    loader = DataLoader(config.DATA_DIR)
    images, labels = loader.load_dataset()
    
    visualizer = DataVisualizer()
    visualizer.plot_class_distribution(labels, "Original Distribution", "original_distribution.png")
    
    cleaner = DataCleaner()
    images, labels = cleaner.clean_dataset(images, labels)
    images = cleaner.resize_images(images, target_size=config.IMG_SIZE)
    
    images = np.array(images)
    labels = np.array(labels)
    
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        images, labels, test_size=config.TEST_SIZE, stratify=labels, random_state=config.RANDOM_STATE
    )
    
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=config.RANDOM_STATE
    )
    
    logger.info(f"  Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    logger.info("\n[STEP 2] Augmenting training data...")
    augmentor = DataAugmentor(augmentation_factor=config.AUGMENTATION_FACTOR)
    X_train_aug, y_train_aug = augmentor.augment_training_data(list(X_train), list(y_train), verbose=True)
    X_train_aug = np.array(X_train_aug)
    
    visualizer.plot_class_distribution(y_train_aug, "After Augmentation", "augmented_distribution.png")
    
    logger.info("\n[STEP 3] Extracting features...")
    cnn_extractor = CNNFeatureExtractor(model_name=config.CNN_MODEL)
    
    train_features = cnn_extractor.extract_batch(X_train_aug, verbose=True)
    val_features = cnn_extractor.extract_batch(X_val, verbose=True)
    test_features = cnn_extractor.extract_batch(X_test, verbose=True)
    
    logger.info(f"  Feature dimensions: {train_features.shape[1]}")
    
    logger.info("\n[STEP 4] Training models...")
    
    svm_model = SVMClassifier()
    svm_model.train(train_features, y_train_aug, optimize=True, verbose=True)
    svm_val_results = svm_model.evaluate(val_features, y_val, verbose=True)
    
    knn_model = KNNClassifier()
    knn_model.train(train_features, y_train_aug, optimize=True, verbose=True)
    knn_val_results = knn_model.evaluate(val_features, y_val, verbose=True)
    
    logger.info("\n[STEP 5] Final evaluation...")
    svm_test_results = svm_model.evaluate(test_features, y_test, verbose=True)
    knn_test_results = knn_model.evaluate(test_features, y_test, verbose=True)
    
    if svm_test_results['macro_f1'] >= knn_test_results['macro_f1']:
        best_model_name = 'svm'
        best_results = svm_test_results
    else:
        best_model_name = 'knn'
        best_results = knn_test_results
    
    logger.info("\n[STEP 6] Saving models...")
    svm_model.save()
    knn_model.save()
    
    joblib.dump({'cnn_model': config.CNN_MODEL, 'feature_dim': cnn_extractor.feature_dim}, config.MODELS_DIR / "cnn_extractor_info.joblib")
    joblib.dump({
        'best_model_name': best_model_name,
        'svm_results': {'accuracy': svm_test_results['accuracy'], 'macro_f1': svm_test_results['macro_f1']},
        'knn_results': {'accuracy': knn_test_results['accuracy'], 'macro_f1': knn_test_results['macro_f1']}
    }, config.MODELS_DIR / "training_info.joblib")
    
    evaluator = ModelEvaluator()
    evaluator.plot_confusion_matrix(y_test, best_results['predictions'], best_model_name, save=True)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info(f"Best Model: {best_model_name.upper()}, Macro F1: {best_results['macro_f1']:.4f}")
    logger.info("="*60)
    
    return {'accuracy': best_results['accuracy'], 'macro_f1': best_results['macro_f1'], 'best_model': best_model_name, 'predictions': best_results['predictions']}


def run_realtime():
    from deployment.realtime_classifier import RealtimeClassifier
    classifier = RealtimeClassifier()
    classifier.run()


def run_batch(input_folder, output_file=None):
    from deployment.batch_classifier import BatchClassifier
    classifier = BatchClassifier()
    results = classifier.classify_folder(input_folder)
    if output_file:
        results.to_csv(output_file, index=False)
    return results


def main():
    parser = argparse.ArgumentParser(description='Waste Classification System')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'realtime', 'batch'])
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train_models()
    elif args.mode == 'realtime':
        run_realtime()
    elif args.mode == 'batch':
        if args.input is None:
            print("Error: --input required for batch mode")
            return
        run_batch(args.input, args.output)


if __name__ == "__main__":
    main()
