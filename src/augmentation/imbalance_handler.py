from torch.utils.data import WeightedRandomSampler
import numpy as np
from collections import Counter

def get_weighted_sampler(dataset):
    labels = [dataset[i]['label'] for i in range(len(dataset))]
    class_counts = Counter(labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    weights = [class_weights[label] for label in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
