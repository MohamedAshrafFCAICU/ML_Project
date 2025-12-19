import os
from pathlib import Path

def _detect_environment():
    if os.environ.get('SPACE_ID'):
        return 'huggingface'
    if os.environ.get('STREAMLIT_SHARING_MODE') or os.path.exists('/mount/src'):
        return 'streamlit_cloud'
    if os.environ.get('RENDER'):
        return 'render'
    try:
        import google.colab
        return 'colab'
    except ImportError:
        pass
    if os.path.exists('/kaggle/input'):
        return 'kaggle'
    return 'local'

ENVIRONMENT = _detect_environment()

class Config:
    if ENVIRONMENT == 'huggingface':
        PROJECT_ROOT = Path('/app')
        SAVE_ROOT = PROJECT_ROOT
    elif ENVIRONMENT == 'streamlit_cloud':
        PROJECT_ROOT = Path('/mount/src/ml_project')
        SAVE_ROOT = PROJECT_ROOT
    elif ENVIRONMENT == 'render':
        PROJECT_ROOT = Path(__file__).parent.parent
        SAVE_ROOT = PROJECT_ROOT
    elif ENVIRONMENT == 'colab':
        PROJECT_ROOT = Path("/content/ML_Project")
        SAVE_ROOT = Path("/content/drive/MyDrive/ML_Project_Output")
    elif ENVIRONMENT == 'kaggle':
        PROJECT_ROOT = Path("/kaggle/working/project")
        SAVE_ROOT = Path("/kaggle/working")
    else:
        PROJECT_ROOT = Path(__file__).parent.parent
        SAVE_ROOT = PROJECT_ROOT
    
    DATA_DIR = PROJECT_ROOT / "dataset"
    OUTPUT_DIR = SAVE_ROOT / "output"
    MODELS_DIR = SAVE_ROOT / "saved_models"
    FIGURES_DIR = OUTPUT_DIR / "figures"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    
    if ENVIRONMENT == 'local':
        for dir_path in [OUTPUT_DIR, MODELS_DIR, FIGURES_DIR, REPORTS_DIR, PREDICTIONS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    IMG_SIZE = (224, 224)
    IMG_CHANNELS = 3
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    CLASSES = {
        0: 'cardboard', 1: 'glass', 2: 'metal',
        3: 'paper', 4: 'plastic', 5: 'trash', 6: 'unknown'
    }
    NUM_CLASSES = 7
    PRIMARY_CLASSES = 6
    
    CLASS_COLORS = {
        0: (139, 69, 19), 1: (0, 255, 0), 2: (192, 192, 192),
        3: (255, 255, 0), 4: (0, 0, 255), 5: (128, 128, 128), 6: (255, 0, 255)
    }
    
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    RANDOM_STATE = 42
    
    AUGMENTATION_FACTOR = 2
    HORIZONTAL_FLIP_PROB = 0.5
    VERTICAL_FLIP_PROB = 0.2
    ROTATION_DEGREES = 30
    COLOR_JITTER = {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.15}
    
    CNN_MODEL = 'efficientnet'
    CNN_FEATURE_DIM = 1280
    CNN_BATCH_SIZE = 32
    
    SVM_KERNEL = 'rbf'
    SVM_C = 10
    SVM_GAMMA = 'scale'
    SVM_CLASS_WEIGHT = 'balanced'
    SVM_PARAM_GRID = {'C': [1, 10, 100], 'gamma': ['scale', 0.01], 'kernel': ['rbf']}
    
    KNN_N_NEIGHBORS = 5
    KNN_WEIGHTS = 'distance'
    KNN_METRIC = 'cosine'
    KNN_PARAM_GRID = {'n_neighbors': [3, 5, 7], 'weights': ['distance'], 'metric': ['cosine', 'euclidean']}
    
    CV_FOLDS = 3
    CONFIDENCE_THRESHOLD = 0.5
    PRIMARY_METRIC = 'macro_f1'
    
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    
    @classmethod
    def get_class_name(cls, class_id):
        return cls.CLASSES.get(class_id, 'unknown')
    
    @classmethod
    def get_class_id(cls, class_name):
        for id, name in cls.CLASSES.items():
            if name.lower() == class_name.lower():
                return id
        return 6
    
    @staticmethod
    def get_environment():
        return ENVIRONMENT

config = Config()

print(f"Environment: {ENVIRONMENT}")
print(f"Models Dir: {Config.MODELS_DIR}")
