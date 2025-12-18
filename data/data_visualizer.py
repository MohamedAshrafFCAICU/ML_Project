import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import config


class DataVisualizer:
    def __init__(self, save_dir=None):
        self.save_dir = Path(save_dir) if save_dir else config.FIGURES_DIR
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_class_distribution(self, labels, title="Class Distribution", save_name=None):
        distribution = Counter(labels)
        classes = [config.CLASSES[i] for i in range(config.NUM_CLASSES) if i in distribution]
        counts = [distribution[i] for i in range(config.NUM_CLASSES) if i in distribution]
        
        plt.figure(figsize=(12, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
        bars = plt.bar(classes, counts, color=colors, edgecolor='black')
        
        plt.xlabel('Material Class', fontsize=12)
        plt.ylabel('Number of Samples', fontsize=12)
        plt.title(title, fontsize=14)
        plt.xticks(rotation=45, ha='right')
        
        for bar, val in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, str(val), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / save_name, dpi=150, bbox_inches='tight')
        
        plt.close()
        return distribution
