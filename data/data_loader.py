import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import Counter

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class DataLoader:
    def __init__(self, data_dir=None):
        self.data_dir = Path(data_dir) if data_dir else config.DATA_DIR
        self.images = []
        self.labels = []
        self.file_paths = []
        self.class_counts = {}
        
    def load_dataset(self, verbose=True):
        if verbose:
            print(f"Loading dataset from {self.data_dir}...")
        
        self.images = []
        self.labels = []
        self.file_paths = []
        
        for class_id, class_name in config.CLASSES.items():
            possible_dirs = [
                self.data_dir / class_name.lower(),
                self.data_dir / class_name,
                self.data_dir / str(class_id)
            ]
            
            class_dir = None
            for dir_path in possible_dirs:
                if dir_path.exists():
                    class_dir = dir_path
                    break
            
            if class_dir is None:
                if verbose:
                    print(f"  Warning: No directory found for class '{class_name}'")
                continue
            
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            image_files = []
            for ext in extensions:
                image_files.extend(list(class_dir.glob(ext)))
                image_files.extend(list(class_dir.glob(ext.upper())))
            
            if verbose:
                print(f"  Loading {class_name}: {len(image_files)} images")
            
            for img_path in tqdm(image_files, desc=f"  {class_name}", disable=not verbose):
                img = cv2.imread(str(img_path))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    self.images.append(img)
                    self.labels.append(class_id)
                    self.file_paths.append(str(img_path))
        
        self.labels = np.array(self.labels)
        self.class_counts = Counter(self.labels)
        
        if verbose:
            print(f"\nTotal loaded: {len(self.images)} images")
        
        return self.images, self.labels
    
    def get_class_distribution(self):
        return {config.CLASSES[k]: v for k, v in sorted(self.class_counts.items())}
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
