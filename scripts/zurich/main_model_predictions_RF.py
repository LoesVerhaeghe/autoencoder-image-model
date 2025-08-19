import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path as os_path 
from sklearn.ensemble import RandomForestRegressor
import torch

path_to_mainfolder = "outputs/zurich/all_images_encoded"

all_image_folders = sorted(listdir(path_to_mainfolder))
num_folders_total = len(all_image_folders)

df_TSS=pd.read_csv('data/zurich/SST_TSS.csv', index_col=0)


# --- Load ALL Features, Aggregate, Create Labels and Folder Mapping ---
all_features_list = []
all_labels_TSS_list = []
feature_folder_map = [] # Keep track of which folder index each feature belongs to
processed_folder_indices = [] # Keep track of folders we actually found data for

folder_idx_counter = 0 
for folder in all_image_folders:
    # Path to embeddings for this folder
    path_to_folder = f"{path_to_mainfolder}/{folder}"
    images_in_folder_count = 0
    for subfolder in listdir(path_to_folder):
        if subfolder!='RLB-S' and subfolder!='RLB-N':
            path_to_subfolder=f"{path_to_folder}/{subfolder}"
            if not os_path.exists(path_to_subfolder) or not listdir(path_to_subfolder):
                print(f"Warning: Embeddings path not found or empty for folder {folder}/{subfolder}, skipping.")
                continue

            images_list_embeddings = listdir(path_to_subfolder)
            for image_file in images_list_embeddings:
                try:
                    img_path = f"{path_to_subfolder}/{image_file}"
                    embedding = torch.load(img_path).cpu().numpy() 
                    all_features_list.append(embedding)

                    # Assign labels corresponding to this FOLDER (time point 'folder_idx_counter')
                    all_labels_TSS_list.append(df_TSS['SST_TSS'].loc[folder].item())
                    images_in_folder_count += 1
                except Exception as e:
                    print(f"Error loading or processing {img_path}: {e}")

                feature_folder_map.append(folder_idx_counter) # Store the *processed* folder index
    if images_in_folder_count > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed

    folder_idx_counter += 1 # Move to the next folder's error value

all_features_agg = np.array(all_features_list) # Shape (num_total_images, num_agg_features)
TSS_labels = np.array(all_labels_TSS_list)
feature_folder_map = np.array(feature_folder_map) 

print(f"Total folders found: {num_folders_total}")
print(f"Folders successfully processed: {len(processed_folder_indices)}")
print(f"Total image features loaded: {len(all_features_agg)}")
print(f"Labels array length: {len(TSS_labels)}")
print(f"Feature matrix shape: {all_features_agg.shape}")
assert len(all_features_agg) == len(TSS_labels) # Check features match labels

# --- Split Based on Time (Processed Folders) ---
# Use the number of *processed* folders for splitting
train_indices_folders=np.arange(0, 210)
val_indices_folders = np.arange(210, 260) 
test_indices_folders=np.arange(260, 379)

# Get indices of features belonging to train folders vs test folders
train_indices = np.where(np.isin(feature_folder_map, train_indices_folders))[0]
val_indices = np.where(np.isin(feature_folder_map, val_indices_folders))[0]
test_indices = np.where(np.isin(feature_folder_map, test_indices_folders))[0]

# --- Create Train ---
X_train = all_features_agg[train_indices]
y_train = TSS_labels[train_indices]
X_val = all_features_agg[val_indices]
y_val = TSS_labels[val_indices]

############################################################
#### Train model (Random Forest) ####
############################################################


#### hyperparametertune

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
param_dist = {
    'n_estimators': [50, 100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', None]
}
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
random_search = RandomizedSearchCV(
    rf,
    param_distributions=param_dist,
    n_iter=20,  
    cv=[(train_indices, val_indices)],  # time-aware split
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
random_search.fit(all_features_agg, TSS_labels)
print("Best params:", random_search.best_params_)
best_rf = random_search.best_estimator_

# --- Predict on ALL Data for Hybrid Model ---
y_pred_all = best_rf.predict(all_features_agg)

# --- Average Predictions and Calculate Std Dev per Time Point (Folder) ---
average_preds = []
std_dev = []
i=0
for folder in all_image_folders:
    # Path to embeddings for this folder
    path_to_folder = f"{path_to_mainfolder}/{folder}"
    images_in_folder_count = 0
    temporary_pred=[]
    for subfolder in listdir(path_to_folder):
        if subfolder!='RLB-S' and subfolder!='RLB-N':
            path_to_subfolder=f"{path_to_folder}/{subfolder}"
            images_list_embeddings = listdir(path_to_subfolder)
            for image_file in images_list_embeddings:
                temporary_pred.append(y_pred_all[i])
                i += 1 
    average_preds.append(np.median(temporary_pred)) 
    std_dev.append(np.std(temporary_pred))

all_image_folders_datetime=pd.to_datetime(all_image_folders)
df_TSS.index=pd.to_datetime(df_TSS.index)

# --- Construct Model preds and Uncertainty Bands ---
# Ensure index alignment - crucial if folders were skipped
TSS_predictions = pd.Series(
    average_preds,
    index=all_image_folders_datetime
)

TSS_upper = pd.Series(
    TSS_predictions.values + std_dev,
    index=all_image_folders_datetime
)
TSS_lower = pd.Series(
    TSS_predictions.values - std_dev,
    index=all_image_folders_datetime
)

# --- Plotting Results ---

plt.figure(figsize=(14, 3), dpi=200)
plt.rcParams.update({'font.size': 12})    
plt.plot(df_TSS['SST_TSS'], '.-', label='Measurements', color='blue')
plt.plot(TSS_predictions.iloc[train_indices_folders], '.-', label='HM predictions (train)', color='orange')
plt.plot(TSS_predictions.iloc[val_indices_folders], '.-', label='HM predictions (validation)', color='green')
plt.plot(TSS_predictions.iloc[test_indices_folders], '.-', label='HM predictions (test)', color='red')
plt.fill_between(TSS_predictions.index[train_indices_folders],
                 TSS_lower.iloc[train_indices_folders],
                 TSS_upper.iloc[train_indices_folders],
                 color='orange', alpha=0.2, zorder=1)
plt.fill_between(TSS_predictions.index[val_indices_folders],
                 TSS_lower.iloc[val_indices_folders],
                 TSS_upper.iloc[val_indices_folders],
                 color='green', alpha=0.2, zorder=1)
plt.fill_between(TSS_predictions.index[test_indices_folders],
                 TSS_lower.iloc[test_indices_folders],
                 TSS_upper.iloc[test_indices_folders],
                 color='red', alpha=0.2, zorder=1)
plt.xlabel("Time")
plt.ylabel("TSS (mg/L)")
plt.legend()
plt.show()
