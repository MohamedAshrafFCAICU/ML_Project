import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config

import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset


class ImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = self.images[idx]
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img


class CNNFeatureExtractor:
    def __init__(self, model_name='efficientnet'):
        self.model_name = model_name.lower()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_dim = 1280
        self.input_size = config.IMG_SIZE
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
        
        self._build_model()
        
    def _build_model(self):
        print(f"Loading EfficientNet-B0 as feature extractor...")
        print(f"  Device: {self.device}")
        
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(base_model.features, base_model.avgpool, nn.Flatten())
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"  Feature dimension: {self.feature_dim}")
    
    def extract_features(self, image):
        try:
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(img_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features.flatten()
        except Exception as e:
            print(f"Error extracting features: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def extract_batch(self, images, batch_size=None, verbose=True):
        batch_size = batch_size or config.CNN_BATCH_SIZE
        
        if verbose:
            print(f"\nExtracting EfficientNet features from {len(images)} images...")
        
        dataset = ImageDataset(images, transform=self.transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        
        all_features = []
        
        with torch.no_grad():
            iterator = tqdm(dataloader, desc="Extracting features") if verbose else dataloader
            for batch in iterator:
                batch = batch.to(self.device)
                features = self.model(batch)
                all_features.append(features.cpu().numpy())
        
        features_array = np.vstack(all_features).astype(np.float32)
        
        if verbose:
            print(f"  Feature matrix shape: {features_array.shape}")
        
        return features_array
    
    def get_feature_info(self):
        return {
            'model': 'EfficientNet-B0',
            'feature_dim': self.feature_dim,
            'input_size': self.input_size,
            'device': str(self.device)
        }
