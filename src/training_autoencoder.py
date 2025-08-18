import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt

def train_autoencoder(num_epochs, model, optimizer, device, train_loader, val_loader=None,
                         loss_fn=nn.MSELoss(), skip_epoch_stats=False, plot_losses_path=None, save_model_path=None, 
                         patience=5):
    # Use GPU if available
    model.to(device)

    #log_dict = {'train_loss_per_epoch': [], 'val_loss_per_epoch': []}
    log_dict = {
        'train_total_loss_per_epoch': [],
        'train_mse_loss_per_epoch': [],
        'train_ssim_loss_per_epoch': [],
        'val_total_loss_per_epoch': [],
        'val_mse_loss_per_epoch': [],
        'val_ssim_loss_per_epoch': []
    }
    start_time = time.time()

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    # Training the autoencoder
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()  # Set the model to training mode
        #train_loss = 0  # Initialize epoch loss
        # Initialize epoch accumulators for all loss components
        train_total_loss_accum = 0
        train_mse_loss_accum = 0
        train_ssim_loss_accum = 0

        for batch in train_loader:
            # Move the batch to the device
            batch_data = batch.to(device)

            # Forward pass
            outputs = model(batch_data)
            #loss = loss_fn(outputs, batch_data)
            total_loss, mse_loss, ssim_loss = loss_fn(outputs, batch_data, device)

            # Backward pass and optimization
            total_loss.backward()  # Compute gradients
            optimizer.step()  # Update parameters
            optimizer.zero_grad(set_to_none=True)  # Reset gradients

            # Accumulate the loss for the current epoch
            #train_loss += loss.item()
            # --- Accumulate losses for logging ---
            train_total_loss_accum += total_loss.item()
            train_mse_loss_accum += mse_loss.item()
            train_ssim_loss_accum += ssim_loss.item()

        #avg_train_loss = train_loss / len(train_loader) # train loss for this epoch
        # Calculate average losses for the epoch
        avg_train_total_loss = train_total_loss_accum / len(train_loader)
        avg_train_mse_loss = train_mse_loss_accum / len(train_loader)
        avg_train_ssim_loss = train_ssim_loss_accum / len(train_loader)

        #log_dict['train_loss_per_epoch'].append(avg_train_loss)
        # Log training losses
        log_dict['train_total_loss_per_epoch'].append(avg_train_total_loss)
        log_dict['train_mse_loss_per_epoch'].append(avg_train_mse_loss)
        log_dict['train_ssim_loss_per_epoch'].append(avg_train_ssim_loss)

        # Validation loop (if validation loader is provided)
        avg_val_total_loss = float('nan') # Use NaN if no validation
        avg_val_mse_loss = float('nan')
        avg_val_ssim_loss = float('nan')

        if val_loader is not None:
            model.eval()
            val_total_loss_accum = 0
            val_mse_loss_accum = 0
            val_ssim_loss_accum = 0
            with torch.no_grad(): # disable gradient computation during validation
                for batch in val_loader:
                    batch_data=batch.to(device)
                    outputs=model(batch_data)
                    total_loss, mse_loss, ssim_loss = loss_fn(outputs, batch_data, device)
                    # Accumulate validation losses
                    val_total_loss_accum += total_loss.item()
                    val_mse_loss_accum += mse_loss.item()
                    val_ssim_loss_accum += ssim_loss.item()
                    #loss=loss_fn(outputs, batch_data)
                    #val_loss += loss.item()
            
            # Calculate average validation losses
            avg_val_total_loss = val_total_loss_accum / len(val_loader)
            avg_val_mse_loss = val_mse_loss_accum / len(val_loader)
            avg_val_ssim_loss = val_ssim_loss_accum / len(val_loader)

            # Log validation losses
            log_dict['val_total_loss_per_epoch'].append(avg_val_total_loss)
            log_dict['val_mse_loss_per_epoch'].append(avg_val_mse_loss)
            log_dict['val_ssim_loss_per_epoch'].append(avg_val_ssim_loss)
            #avg_val_loss= val_loss / len(val_loader) # val loss for this epoch
            #log_dict['val_loss_per_epoch'].append(avg_val_loss)

            # Early stopping logic
            if avg_val_total_loss  < best_val_loss:
                best_val_loss = avg_val_total_loss 
                patience_counter = 0  # Reset the counter when improvement occurs
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}. Best validation loss: {best_val_loss:.4f}")
                    break  # Stop training

        if not skip_epoch_stats:
            print(f'Epoch [{epoch + 1}/{num_epochs}] | Time: {((time.time() - epoch_start_time)/60):.2f} min')
            print(f'  Train Loss: Total={avg_train_total_loss:.4f} (MSE={avg_train_mse_loss:.4f}, SSIM={avg_train_ssim_loss:.4f})')
            if val_loader is not None:
                print(f'  Val Loss  : Total={avg_val_total_loss:.4f} (MSE={avg_val_mse_loss:.4f}, SSIM={avg_val_ssim_loss:.4f})')
            else:
                print()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))

    if plot_losses_path is not None:
        plt.figure()
        plt.plot(log_dict['train_total_loss_per_epoch'], '.-', label='Total train loss')
        plt.plot(log_dict['train_mse_loss_per_epoch'], '.-', label='MSE train loss')
        plt.plot(log_dict['train_ssim_loss_per_epoch'], '.-', label='SSIM train loss')
        plt.plot(log_dict['val_total_loss_per_epoch'], '.-', label='Total val loss')
        plt.plot(log_dict['val_mse_loss_per_epoch'], '.-', label='MSE val loss')
        plt.plot(log_dict['val_ssim_loss_per_epoch'], '.-', label='SSIM val loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{plot_losses_path}', dpi=300, bbox_inches='tight', pad_inches=0.1)  
        plt.show()

    if save_model_path is not None:
        torch.save(model, save_model_path)

    return model, log_dict  # Return the trained model

