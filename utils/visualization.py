import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class Visualization:
    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else config.FIGURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_images_grid(self, images, titles=None, cols=5, figsize=None, save_name=None):
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        if figsize is None:
            figsize = (3 * cols, 3 * rows)
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten()
        
        for i, ax in enumerate(axes):
            if i < n_images:
                ax.imshow(images[i])
                if titles and i < len(titles):
                    ax.set_title(titles[i], fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def plot_confidence_distribution(self, confidences, predictions=None, save_name=None):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(confidences, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=config.CONFIDENCE_THRESHOLD, color='r', linestyle='--', label=f'Threshold ({config.CONFIDENCE_THRESHOLD})')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.legend()
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        plt.close()
