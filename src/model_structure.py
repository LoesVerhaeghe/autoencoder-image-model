import torch
import torch.nn as nn

# Defining Autoencoder model
class Autoencoder(nn.Module):
   def __init__(self):
       super().__init__()

       self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 32, kernel_size=3, stride=2, padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 8, kernel_size=3, stride=2, padding=1),  
            nn.LeakyReLU(),
            nn.BatchNorm2d(8),   
            nn.Flatten(),  # Flatten before latent space
            nn.Linear(8*32*24, 1024), 
            nn.ReLU() 
        )

       self.decoder = nn.Sequential(
            nn.Linear(1024, 8*32*24),  # Match the flattened dimension
            nn.ReLU(),
            nn.Unflatten(1, (8, 24, 32)),  # Reshape to match the input to ConvTranspose
            nn.ConvTranspose2d(8, 32, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.LeakyReLU(),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.LeakyReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), 
            nn.LeakyReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1), # Out: (B, 1, H, W)
            nn.Sigmoid() 
        )

       
   def forward(self, x, get_encoded=False):
        x = self.encoder(x)
        
        if get_encoded:  # If flag is True, return the encoded images
           return x

        x = self.decoder(x)
        return x
