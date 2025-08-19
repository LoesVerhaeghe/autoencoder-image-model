from utils.helpers import extract_image_paths_zurich
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from PIL import Image as PImage


class ContrastiveMicroscopicImagesZurich(Dataset):
    def __init__(self, root,  start_folder, end_folder, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.images = extract_image_paths_zurich(root, start_folder=start_folder, end_folder=end_folder)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = torch.load(image_path).to(torch.float32)

        if image is None:
            print(f"Failed to load image at index {idx} (path: {image_path})")
            return None

        if self.transform:
            try:
                x1 = self.transform(image)
                x2 = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None
        else:
            x1=image
            x2=image

        return x1, x2