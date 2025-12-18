import os
import shutil
from pathlib import Path
import json
import pickle
import joblib

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class FileUtils:
    @staticmethod
    def ensure_dir(path):
        Path(path).mkdir(parents=True, exist_ok=True)
        return Path(path)
    
    @staticmethod
    def get_image_files(directory, recursive=False):
        directory = Path(directory)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        
        image_files = []
        for ext in extensions:
            if recursive:
                image_files.extend(list(directory.rglob(ext)))
                image_files.extend(list(directory.rglob(ext.upper())))
            else:
                image_files.extend(list(directory.glob(ext)))
                image_files.extend(list(directory.glob(ext.upper())))
        
        return sorted(image_files)
    
    @staticmethod
    def save_json(data, path):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @staticmethod
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def save_joblib(data, path):
        joblib.dump(data, path)
    
    @staticmethod
    def load_joblib(path):
        return joblib.load(path)
    
    @staticmethod
    def clean_directory(directory, keep_dir=True):
        directory = Path(directory)
        if directory.exists():
            shutil.rmtree(directory)
        if keep_dir:
            directory.mkdir(parents=True, exist_ok=True)
