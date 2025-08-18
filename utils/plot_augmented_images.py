import matplotlib.pyplot as plt
from PIL import Image
import kornia.augmentation as K
import torch

augment = torch.nn.Sequential(
    K.RandomHorizontalFlip(p=0.5),
    K.RandomVerticalFlip(p=0.5),
    K.RandomRotation(degrees=10.0, resample='bilinear'),
    K.RandomAffine(degrees=0, translate=(0.05, 0.05)),
   # K.CenterCrop((256, 256))
)

image_path = 'data/microscope_images_CLAHE/2023-11-16/basin5/10x/2023-11-16 Pilote 10X B5 1.pt'
original_image = torch.load(image_path)

fig, axs = plt.subplots(2, 3, figsize=(12, 6))  
axs = axs.flatten()
# Show original image in the first subplot
axs[0].imshow(original_image.squeeze(0).squeeze(0), cmap='gray')
axs[0].axis('off')
axs[0].set_title('Original')

# Show 5 augmentations
for i in range(5):
    augmented_image = augment(original_image)
    axs[i + 1].imshow(augmented_image.squeeze(0).squeeze(0), cmap='gray')
    axs[i + 1].axis('off')
    axs[i + 1].set_title(f'Augmentation {i+1}')

plt.tight_layout()
plt.show()