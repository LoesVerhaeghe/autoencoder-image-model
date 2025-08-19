import matplotlib.pyplot as plt
import torch


# Function to display original and reconstructed images
def visualize_reconstruction(model, device, test_loader, num_images=5, path='outputs'):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for testing/inference
        for batch in test_loader:
            batch = batch.to(device)

            # Forward pass
            batch_recon = model(batch)
            batch = batch.cpu()
            batch_recon = batch_recon.cpu()
            
            fig, axes = plt.subplots(2, num_images, figsize=(5*num_images, 10))
            for i in range(num_images):
                # Original images
                axes[0, i].imshow(batch[i].permute(1, 2, 0).numpy(), cmap='grey')  # Convert CHW to HWC
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstructed images
                axes[1, i].imshow(batch_recon[i].permute(1, 2, 0).numpy(), cmap='grey')  # Convert CHW to HWC
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(f'{path}', dpi=300, bbox_inches='tight', pad_inches=0.1)  
            plt.close()  
            break 


# Function to display original and reconstructed images
def visualize_reconstruction_contrastive(model, device, test_loader, num_images=5, path='outputs'):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for testing/inference
        for batch in test_loader:
            x1 , x2 = batch
            x1 = x1.to(device)

            # Forward pass
            x1_recon = model(x1)
            x1 = x1.cpu()
            x1_recon = x1_recon.cpu()
            
            fig, axes = plt.subplots(2, num_images, figsize=(5*num_images, 10))
            for i in range(num_images):
                # Original images
                axes[0, i].imshow(x1[i].permute(1, 2, 0).numpy(), cmap='grey')  # Convert CHW to HWC
                axes[0, i].set_title('Original')
                axes[0, i].axis('off')
                
                # Reconstructed images
                axes[1, i].imshow(x1_recon[i].permute(1, 2, 0).numpy(), cmap='grey')  # Convert CHW to HWC
                axes[1, i].set_title('Reconstructed')
                axes[1, i].axis('off')

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(f'{path}', dpi=300, bbox_inches='tight', pad_inches=0.1)  
            plt.close()  
            break 
