from utils.plotting_utilities import visualize_reconstruction_contrastive
from src.model_structure_contrastive import AutoencoderWithContrastiveHead
from src.training_contrastive_autoencoder import train_contrastive_autoencoder
from src.images_dataset_contrastive import ContrastiveMicroscopicImagesZurich, MicroscopicImagesZurich
import torch.optim as optim
from torchsummary import summary
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import shutil
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class NTXentLoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Compute the NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

        Args:
            z1 (torch.Tensor): Batch of embeddings from view 1, shape (B, D)
            z2 (torch.Tensor): Batch of embeddings from view 2, shape (B, D)

        Returns:
            torch.Tensor: scalar loss
        """
        batch_size = z1.size(0)

        # Normalize embeddings to unit length (important for cosine similarity)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Concatenate to get 2B x D
        # First B rows are z1, next B rows are z2
        z = torch.cat([z1, z2], dim=0)

        # Compute similarity matrix for all pairs (2B x 2B)
        # Each element sim[i,j] = cosine similarity(z[i], z[j]) / temperature
        sim = torch.matmul(z, z.T) / self.temperature

        # For each sample i, the similarity with itself (i==j) is not valid, set to large negative to exclude
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -float('inf'))

        # For the 2B samples, define the indices of positive samples:
        # For i in [0 .. B-1], positive is i+B
        # For i in [B .. 2B-1], positive is i-B
        positives = torch.arange(batch_size, device=z.device)
        positive_indices = torch.cat([positives + batch_size, positives])

        # Compute loss
        # loss encourages the similarity of the positive pair to be higher than all other pairs
        loss = F.cross_entropy(sim, positive_indices)

        return loss

#use GPU if available
torch.cuda.set_device(0) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', DEVICE)

# ### Loss functions
mse_loss_fn = nn.MSELoss()
ssim_metric = SSIM(data_range=1.0, gaussian_kernel=True, kernel_size=7).to(DEVICE)
contrastive_loss_fn = NTXentLoss(temperature=0.5)

def combined_loss(x1, z1, recon1, x2, z2, recon2):
    mse1 = 10 * mse_loss_fn(recon1, x1) 
    mse2 = 10 * mse_loss_fn(recon2, x2)
    mse_loss = (mse1 + mse2) / 2

    ssim_val1 = ssim_metric(recon1, x1)
    ssim_val2 = ssim_metric(recon2, x2)
    ssim1 = 1.0 - ssim_val1
    ssim2 = 1.0 - ssim_val2
    ssim_loss = (ssim1 + ssim2) / 2

    # Contrastive loss
    contrastive_loss = contrastive_loss_fn(z1, z2)

    # Total loss
    total_loss = mse_loss + ssim_loss + contrastive_loss

    return total_loss, mse_loss, ssim_loss, contrastive_loss


transform = transforms.Compose([
    #color transformations
    transforms.RandomApply(
        torch.nn.ModuleList([transforms.ColorJitter(
            brightness=0.4, contrast=0, saturation=0, hue=0),
            ]), p=0.5),        

    #geometric transformations
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    #transforms.RandomApply(
    #    torch.nn.ModuleList([transforms.RandomRotation(180),
    #        ]), p=0.5),        
    #transforms.RandomErasing(p=0.5, scale=(0.02, 0.2)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
])

# load model structure
model=AutoencoderWithContrastiveHead().to(DEVICE)
print(summary(model, input_size=(1, 512, 384)))

# Setting training parameters
RANDOM_SEED= 69
LEARNING_RATE = 1e-3 
BATCH_SIZE = 8
NUM_EPOCHS = 10000 # we use early stopping anyway
OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)
save_model_path='outputs/model_contrastiveautoencoder.pt'
torch.manual_seed(RANDOM_SEED)
SCHEDULER = ReduceLROnPlateau(
    OPTIMIZER,
    mode='min',      # The scheduler will reduce the LR when the quantity monitored has stopped decreasing.
    factor=0.5,      # Factor by which the learning rate will be reduced. new_lr = lr * factor.
    patience=3     # Number of epochs with no improvement after which learning rate will be reduced.
)

### load data
base_folder = "data/Mikroskopie_structured_CLAHE"
dataset = ContrastiveMicroscopicImagesZurich(root=base_folder, start_folder='2010-03-10', end_folder='2020-12-14', transform=transform)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ### Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True, shuffle=False)

# Train autoencoder
# trained_model, log_dict=train_contrastive_autoencoder(NUM_EPOCHS, 
#                                           model, 
#                                           OPTIMIZER, 
#                                           DEVICE, 
#                                           train_loader, 
#                                           val_loader, 
#                                           loss_fn=combined_loss, 
#                                           scheduler=SCHEDULER,
#                                           skip_epoch_stats=False, 
#                                           plot_losses_path='outputs/losses.png', 
#                                           save_model_path=save_model_path)

model = torch.load(save_model_path, map_location=DEVICE)
visualize_reconstruction_contrastive(model, DEVICE, train_loader, num_images=5, path='outputs/traindataset_reconstruction.png')
visualize_reconstruction_contrastive(model, DEVICE, val_loader, num_images=5, path='outputs/validationdataset_reconstruction.png')

# # #to generate encoded images with trained model, first copy the preprocessed data folder in the output folder and rename, then run this code:

# Copy the folder structure from src_folder to dst_folder
src_folder = base_folder
dst_folder = 'outputs/all_images_encoded_contrastive'
shutil.copytree(src_folder, dst_folder)

dataset= MicroscopicImagesZurich(root=dst_folder,  start_folder='2010-03-10', end_folder='2024-11-28', transform=None)

def encode_images(trained_model, dataset):
    # Initialize dataset and dataloader for the given magnification
    data_loader = DataLoader(dataset, batch_size=1, num_workers=10, pin_memory=True, shuffle=False)
    model.eval()
    
    with torch.no_grad():
        for i, (image_data) in enumerate(data_loader):
            image_data = image_data.to(DEVICE)
            encoded_image = trained_model(image_data, get_encoded=True)
            encoded_image = encoded_image.squeeze(0)  # Remove batch dimension

            # Get the original filename and folder path from the dataset
            original_path = dataset.images[i]  # Adjust based on your dataset class
            folder_path, file_name = os.path.split(original_path)
            new_file_name = file_name.replace(".pt", "_encoded.pt")  # Change extension to .pt

            # Save the encoded tensor and delete the original file
            new_file_path = os.path.join(folder_path, new_file_name)
            torch.save(encoded_image.cpu(), new_file_path)

            if os.path.exists(original_path):
                os.remove(original_path)  # Delete the original file
                
# # Encode and save for both magnifications
encode_images(model, dataset)

# ## plot latent space
# import matplotlib.pyplot as plt
# path='outputs/microscope_images_encoded/2023-10-20/basin5/10x/2023-10-20 Pilote 10X B5 1_encoded.pt'
# #path='outputs/microscope_images_encoded_jointloss/2024-03-12/basin5/40x/12125430_encoded.pt'
# image=torch.load(path)

# # Set up the figure and subplots
# num_channels = image.shape[0]  # Number of filters
# fig, axes = plt.subplots(num_channels, 1, figsize=(5, num_channels * 3))
# for i in range(num_channels):
#     axes[i].imshow(image[i, :, :].numpy(), cmap='gray')
#     axes[i].set_title(f'Filter {i+1}')
#     axes[i].axis('off')
# plt.tight_layout()
# plt.show()