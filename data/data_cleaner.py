import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class DataCleaner:
    def __init__(self):
        self.removed_indices = []
        self.quality_report = {}
        
    def check_image_quality(self, image):
        issues = []
        if image.shape[0] < 32 or image.shape[1] < 32:
            issues.append("too_small")
        if np.std(image) < 5:
            issues.append("uniform")
        if np.mean(image) < 5:
            issues.append("too_dark")
        elif np.mean(image) > 250:
            issues.append("too_bright")
        if len(image.shape) != 3 or image.shape[2] != 3:
            issues.append("invalid_channels")
        return len(issues) == 0, issues
    
    def detect_blur(self, image, threshold=100):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < threshold, laplacian_var
    
    def clean_dataset(self, images, labels, verbose=True):
        if verbose:
            print("Cleaning dataset...")
        
        clean_images = []
        clean_labels = []
        self.removed_indices = []
        
        quality_issues = {'too_small': 0, 'uniform': 0, 'too_dark': 0, 'too_bright': 0, 'invalid_channels': 0, 'severely_blurry': 0}
        
        iterator = tqdm(enumerate(zip(images, labels)), total=len(images), disable=not verbose, desc="Cleaning")
        
        for i, (img, label) in iterator:
            is_valid, issues = self.check_image_quality(img)
            
            if not is_valid:
                self.removed_indices.append(i)
                for issue in issues:
                    quality_issues[issue] = quality_issues.get(issue, 0) + 1
                continue
            
            is_blurry, blur_score = self.detect_blur(img, threshold=20)
            if is_blurry:
                self.removed_indices.append(i)
                quality_issues['severely_blurry'] += 1
                continue
            
            clean_images.append(img)
            clean_labels.append(label)
        
        self.quality_report = {
            'original_count': len(images),
            'removed_count': len(self.removed_indices),
            'clean_count': len(clean_images),
            'issues': quality_issues
        }
        
        if verbose:
            print(f"\n  Original: {len(images)}, Removed: {len(self.removed_indices)}, Clean: {len(clean_images)}")
        
        return clean_images, np.array(clean_labels)
    
    def resize_images(self, images, target_size=None, verbose=True):
        target_size = target_size or config.IMG_SIZE
        
        if verbose:
            print(f"Resizing images to {target_size}...")
        
        resized = []
        iterator = tqdm(images, disable=not verbose, desc="Resizing")
        
        for img in iterator:
            resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            resized.append(resized_img)
        
        return np.array(resized)
