from utils.helpers import extract_image_paths_zurich
import cv2
import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import shutil 
import numpy as np
from skimage import io
from skimage.morphology import disk
from skimage.filters import rank
# copy the folder containing all images

src_folder = 'data/zurich/Mikroskopie_structured'
dst_folder = 'data/zurich/Mikroskopie_structured_CLAHE'

#shutil.copytree(src_folder, dst_folder)


def preprocess_image(path, target_size=(512, 384)):
    """
    Preprocess an image:
    - Makes sure the longest side is first (landscape mode)
    - Resizes while maintaining aspect ratio
    - Pads with reflection to reach target size
    - Converts to grayscale

    Args:
        path (str): Path to the image file
        target_size (tuple): (width, height)

    Returns:
        np.ndarray: Preprocessed grayscale image of shape (target_height, target_width)
    """
    # Load image
    image = cv2.imread(path)

    if image is None:
        raise ValueError(f"Failed to read image from {path}")

    # Ensure landscape orientation (longest side first)
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = image.shape[:2]

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute scaling factor to maintain aspect ratio
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding amounts
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Apply reflection padding
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT
    )
    return padded



def preprocess_image_CLAHE(path, target_size=(512, 384)):
    # Load image
    image = cv2.imread(path)

    # Ensure landscape orientation (longest side first)
    h, w = image.shape[:2]
    if h > w:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        h, w = image.shape[:2]

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute scaling factor to maintain aspect ratio
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Compute padding amounts
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # Apply reflection padding
    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_REFLECT
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(padded)

    return enhanced



# save all paths
paths=extract_image_paths_zurich(dst_folder, start_folder='2010-03-10', end_folder='2010-03-19')
    
for path in paths:
    # image = cv2.imread(path)
    print(path)
    padded=preprocess_image_CLAHE(path)

    # # Visualize the original and edge-detected images side by side
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=75)

    # ax[0].imshow(image )
    # ax[0].axis('off')

    # ax[1].imshow(padded, cmap='gray')
    # ax[1].axis('off')

    # plt.show()

    transform = transforms.Compose([
    transforms.ToTensor()]) # also normalized to 0-1

    tensor = transform(padded)

    new_path = os.path.splitext(path)[0] + '.pt'

    # save in new folder
    torch.save(tensor, new_path)

    #delete old image
    os.remove(path)