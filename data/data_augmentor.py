import numpy as np
import cv2
from tqdm import tqdm
from collections import Counter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class DataAugmentor:
    def __init__(self, augmentation_factor=None):
        self.augmentation_factor = augmentation_factor or config.AUGMENTATION_FACTOR
        self.augmentation_stats = {}
        
    def horizontal_flip(self, image, prob=None):
        prob = prob or config.HORIZONTAL_FLIP_PROB
        if np.random.random() < prob:
            return cv2.flip(image, 1)
        return image
    
    def vertical_flip(self, image, prob=None):
        prob = prob or config.VERTICAL_FLIP_PROB
        if np.random.random() < prob:
            return cv2.flip(image, 0)
        return image
    
    def random_rotation(self, image, max_degrees=None):
        max_degrees = max_degrees or config.ROTATION_DEGREES
        angle = np.random.uniform(-max_degrees, max_degrees)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    def color_jitter(self, image, params=None):
        params = params or config.COLOR_JITTER
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        
        brightness = 1 + np.random.uniform(-params['brightness'], params['brightness'])
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
        
        saturation = 1 + np.random.uniform(-params['saturation'], params['saturation'])
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        
        hue_shift = np.random.uniform(-params['hue'] * 180, params['hue'] * 180)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        contrast = 1 + np.random.uniform(-params['contrast'], params['contrast'])
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        image = np.clip((image - mean) * contrast + mean, 0, 255).astype(np.uint8)
        
        return image
    
    def random_zoom(self, image, scale_range=(0.8, 1.2)):
        h, w = image.shape[:2]
        scale = np.random.uniform(*scale_range)
        
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(image, (new_w, new_h))
        
        if scale > 1:
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            result = resized[start_h:start_h+h, start_w:start_w+w]
        else:
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = cv2.copyMakeBorder(resized, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)
        
        return cv2.resize(result, (w, h))
    
    def gaussian_noise(self, image, prob=0.3, std=10):
        if np.random.random() < prob:
            noise = np.random.normal(0, std, image.shape).astype(np.float32)
            noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
            return noisy.astype(np.uint8)
        return image
    
    def random_blur(self, image, prob=0.2):
        if np.random.random() < prob:
            ksize = np.random.choice([3, 5])
            return cv2.GaussianBlur(image, (ksize, ksize), 0)
        return image
    
    def augment_image(self, image):
        aug_img = image.copy()
        aug_img = self.horizontal_flip(aug_img)
        aug_img = self.random_rotation(aug_img)
        aug_img = self.color_jitter(aug_img)
        aug_img = self.vertical_flip(aug_img)
        aug_img = self.random_zoom(aug_img)
        aug_img = self.gaussian_noise(aug_img)
        aug_img = self.random_blur(aug_img)
        return aug_img
    
    def augment_training_data(self, images, labels, verbose=True):
        original_count = len(images)
        
        if verbose:
            print(f"\nAugmenting data ({self.augmentation_factor}x)...")
        
        augmented_images = list(images)
        augmented_labels = list(labels)
        
        pbar = tqdm(range(len(images)), desc="Augmenting") if verbose else range(len(images))
        
        for i in pbar:
            img = images[i]
            label = labels[i]
            
            for _ in range(self.augmentation_factor - 1):
                aug_img = self.augment_image(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        if verbose:
            print(f"  Total images: {len(augmented_images)}")
        
        return augmented_images, np.array(augmented_labels)
