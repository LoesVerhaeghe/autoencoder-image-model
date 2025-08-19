from utils.helpers import extract_image_paths_zurich, extract_image_paths_pileaute
from torch.utils.data import Dataset
import torch
import os
import pandas as pd
from PIL import Image as PImage

class MicroscopicImagesZurich(Dataset):
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
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        return image

class MicroscopicImagesPileaute(Dataset):
    def __init__(self, root, start_folder, end_folder, magnification, transform=None):
        """
        Args:
            root (str): Path to the root directory containing images.
            magnification (str or int): Magnification filter used in image path extraction.
            image_type (str): type of dataset that needs to be extracted: 'all', 'old', or 'new', train or test
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.transform = transform
        self.image_paths = extract_image_paths_pileaute(root, start_folder=start_folder, end_folder=end_folder, magnification=magnification)
        self.targets = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = torch.load(image_path).to(torch.float32)
        if image is None:
            print(f"Failed to load image at index {idx} (path: {image_path})")
            return None

        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Transform failed for image at index {idx}: {e}")
                return None

        return image