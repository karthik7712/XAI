# import os
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset
#
# class SingleInputDataset(Dataset):
#     """
#     Dataset that loads smoothed images and corresponding labels from a given directory.
#     Assumes the structure:
#         - data_dir/
#             - smoothed_images/
#             - labels.csv
#     """
#
#     def __init__(self, data_dir, transform=None):
#         self.smoothed_dir = os.path.join(data_dir, 'smoothed_images')
#         self.labels_df = pd.read_csv(os.path.join(data_dir, 'labels.csv'))
#         self.transform = transform
#
#         self.image_names = self.labels_df['image'].tolist()
#         self.labels = self.labels_df['class'].tolist()
#
#     def __len__(self):
#         return len(self.image_names)
#
#     def __getitem__(self, idx):
#         image_name = self.image_names[idx]
#         label = self.labels[idx]
#
#         image_path = os.path.join(self.smoothed_dir, image_name)
#         image = Image.open(image_path).convert('RGB')
#
#         if self.transform:
#             image = self.transform(image)
#
#         return {
#             'smoothed': image,
#             'label': label,
#             'image_name': image_name
#         }

# src/dataset/loader.py

import os
from torch.utils.data import Dataset
from PIL import Image

class SingleInputDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.segmented_dir = os.path.join(root_dir, "segmented_images")
        self.labels_path = os.path.join(root_dir, "labels.csv")
        self.transform = transform

        with open(self.labels_path, 'r') as f:
            lines = f.readlines()[1:]
            self.data = [line.strip().split(',') for line in lines]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, label = self.data[idx]
        img_path = os.path.join(self.segmented_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return {
            'segmented': image,
            'label': label
        }
