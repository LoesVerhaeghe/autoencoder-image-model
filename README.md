# Autoencoder image model

## Overview

This project provides a framework for training autoencoders (including contrastive autoencoders, code in archive since final model uses basic autoencoder) to create a latent reduced space of an image. It also contains scripts to train prediction models (LSTM and Random Forest) on the reduced latent space.

## Project Structure
'''
scripts/                # Main scripts to train autoencoder and prediction models
src/                    # Core modules (datasets, model structures, training routines)
utils/                  # Helper functions for preprocessing, plotting, and data handling
archive/                # Older versions and experimental scripts, e.g. contrastive autoencoder
'''

## Usage
### Training an Autoencoder
'''python scripts/zurich/main_autoencoder.py'''
This script trains a standard autoencoder on your dataset.

### Training a Contrastive Autoencoder
'''python archive/main_autoencoder_contrastive.py'''
Use this for contrastive learning on images.

### Prediction with LSTM
'''python scripts/zurich/main_model_predictions_LSTM.py'''
Use the trained encodings as input to an LSTM for time-series predictions.

### Prediction with Random Forest
'''python scripts/zurich/main_model_predictions_RF.py'''
Use the trained encodings as input to an RF for time-series predictions.
