from utils.plotting_utilities import visualize_reconstruction
from src.model_structure import Autoencoder
from src.training_autoencoder import train_autoencoder
from src.images_dataset import MicroscopicImagesZurich
import torch.optim as optim
from torchsummary import summary
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import shutil
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

#use GPU if available
torch.cuda.set_device(0) 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', DEVICE)

# ### Loss functions
mse_loss_fn = nn.MSELoss()
ssim_metric = SSIM(data_range=1.0, gaussian_kernel=True, kernel_size=7)

def combined_loss(reconstructed, original, device):
    ssim_metric.to(device)
    mse_loss = 10*mse_loss_fn(reconstructed, original)
    ssim_val = ssim_metric(reconstructed, original)
    ssim_loss = 1.0 - ssim_val
    total_loss = mse_loss + ssim_loss
    return total_loss, mse_loss, ssim_loss

# load model structure
model=Autoencoder().to(DEVICE)
print(summary(model, input_size=(1, 512, 384)))

# Setting training parameters
RANDOM_SEED= 69
LEARNING_RATE = 0.0001 # bigger than 0.0001 ends in local minima
BATCH_SIZE = 16
NUM_EPOCHS = 10000 # we use early stopping anyway
OPTIMIZER = optim.Adam(model.parameters(), lr=LEARNING_RATE)
save_model_path='outputs/zurich/model_basicautoencoder.pt'
torch.manual_seed(RANDOM_SEED)
SCHEDULER = ReduceLROnPlateau(
    OPTIMIZER,
    mode='min',      # The scheduler will reduce the LR when the quantity monitored has stopped decreasing.
    factor=0.5,      # Factor by which the learning rate will be reduced. new_lr = lr * factor.
    patience=3     # Number of epochs with no improvement after which learning rate will be reduced.
)

### load data
base_folder = "data/zurich/Mikroskopie_structured_CLAHE"
dataset = MicroscopicImagesZurich(root=base_folder, start_folder='2010-03-10', end_folder='2020-12-14', transform=None)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size 
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# ### Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=1, pin_memory=True, shuffle=False)

# ## Train autoencoder
# trained_model, log_dict=train_autoencoder(NUM_EPOCHS, 
#                                           model, 
#                                           OPTIMIZER, 
#                                           DEVICE, 
#                                           train_loader, 
#                                           val_loader, 
#                                           loss_fn=combined_loss, 
#                                           scheduler=SCHEDULER,
#                                           skip_epoch_stats=False, 
#                                           plot_losses_path='outputs/zurich/losses.png', 
#                                           save_model_path=save_model_path)

model = torch.load(save_model_path, map_location=DEVICE)
visualize_reconstruction(model, DEVICE, train_loader, num_images=10, path='outputs/zurich/traindataset_reconstruction.png')
visualize_reconstruction(model, DEVICE, val_loader, num_images=10, path='outputs/zurich/validationdataset_reconstruction.png')


# # #to generate encoded images with trained model, first copy the preprocessed data folder in the output folder and rename, then run this code:

# Copy the folder structure from src_folder to dst_folder
src_folder = base_folder
dst_folder = 'outputs/zurich/all_images_encoded'
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

# ## plot latent space, only if latent space is not flattened
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