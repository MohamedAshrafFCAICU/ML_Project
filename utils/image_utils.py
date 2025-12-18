import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class ImageUtils:
    @staticmethod
    def load_image(path, color_mode='rgb'):
        img = cv2.imread(str(path))
        if img is None:
            return None
        if color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif color_mode == 'gray':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    
    @staticmethod
    def save_image(image, path, color_mode='rgb'):
        if color_mode == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image)
    
    @staticmethod
    def resize(image, size=None):
        size = size or config.IMG_SIZE
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def normalize(image):
        return image.astype(np.float32) / 255.0
    
    @staticmethod
    def rotate(image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    @staticmethod
    def flip(image, direction='horizontal'):
        if direction == 'horizontal':
            return cv2.flip(image, 1)
        elif direction == 'vertical':
            return cv2.flip(image, 0)
        return cv2.flip(image, -1)
