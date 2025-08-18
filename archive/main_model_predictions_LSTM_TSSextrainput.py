import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import listdir, path as os_path 
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import torch.optim as optim
from copy import deepcopy

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # for single-GPU

    # For deterministic behavior (might slow things down)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# data aggregeren in matrices (3, 24, 1028), aanvullen met 0 als er geen 20 images per dag zijn
class DailySequenceDataset(Dataset):
    def __init__(self, features_per_day, target_per_day, seq_len=3, n_images=24):
        '''
        features_per_day: dict of date -> list of feature arrays per image
        target_per_day: dict of date -> targetvalue
        '''
        self.seq_len = seq_len
        self.n_images = n_images

        self.dates = sorted(list(set(features_per_day.keys()) & set(target_per_day.keys()))) #only use dates that have both features and a target
        self.features_per_day = features_per_day
        self.targets_per_day = target_per_day

        # Maak geldige sequenties
        self.samples = self.create_sequences()

    def create_sequences(self):
        samples = []

        for i in range(self.seq_len - 1, len(self.dates)):
            sequence_dates = self.dates[i - self.seq_len + 1:i] #cannot use current feature because TSS at that moment will also be added...
            feature_seq = []
            prev_targets=[]

            for date in sequence_dates:
                features = self.features_per_day[date]
                n_features=len(features)
                if n_features >= self.n_images:
                    selected=features[:self.n_images] # Selecteer precies n_images (bijv. eerste of random)
                else:
                    selected=features+[np.zeros(self.features_per_day[date][0].shape)] * (self.n_images-n_features)
                feature_seq.append(np.stack(selected))
                prev_targets.append(self.targets_per_day[date])

            feature_tensor = np.stack(feature_seq)  # shape: (seq_len, n_images, feature_dim)
            target = self.targets_per_day[self.dates[i]]
            samples.append((feature_tensor, prev_targets, target))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        feature_seq, prev_targets, target = self.samples[idx]
        feature_seq = torch.tensor(feature_seq, dtype=torch.float32)
        prev_targets = torch.tensor(prev_targets, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return feature_seq, prev_targets, target

class TransformerDailyAggregator(nn.Module):
    def __init__(self, input_dim=1028, embed_dim=512, n_heads=4, n_layers=2, output_dim=2048):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_projector = nn.Sequential(
            nn.Linear(embed_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        x: (B, N, input_dim) where N = n_images_per_day
        """
        x = self.embedding(x)  # (B, N, embed_dim)
        x = self.transformer(x)  # (B, N, embed_dim)
        x = x.mean(dim=1)  # mean-pool over N images
        return self.output_projector(x)  # (B, output_dim)
    
# 3. LSTM 
class LstmArchitecture(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super(LstmArchitecture, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)               # out: (batch, seq_len, hidden)
        out = out[:, -1, :]                 # Take output at last time step
        return self.fc(out).squeeze()       # Final regression prediction

class FullPipelineModel(nn.Module):
    def __init__(self, 
                 input_dim=1028, 
                 embed_dim=512, 
                 n_heads=4, 
                 n_layers=2, 
                 agg_output_dim=2048, 
                 lstm_hidden=32, 
                 num_layers=1):
        super().__init__()

        # Aggregates 24 images x 1028 features → 1 vector
        self.daily_aggregator = TransformerDailyAggregator(
            input_dim=input_dim,
            embed_dim=embed_dim,
            n_heads=n_heads, 
            n_layers=n_layers,
            output_dim=agg_output_dim
        )

        # LSTM model
        self.lstm_model = LstmArchitecture(
            input_size=agg_output_dim+1,
            hidden_size=lstm_hidden,
            num_layers=num_layers
        )

    def forward(self, x, prev_targets):
        # x: [batch, seq_len=3, n_images=24, feature_dim=1028]
        B, S, N, D = x.shape

        # Flatten to [B*S, N, D] to put into daily aggregator
        x = x.view(B * S, N, D)

        # Aggregate each day → [B*S, 2048]
        aggregated = self.daily_aggregator(x)

        # Reshape back to sequence → [B, S, 2048]
        aggregated_seq = aggregated.view(B, S, -1)
        # print('shape aggregated seq:', aggregated_seq.shape)
        # print('shape prevtargets:', prev_targets.shape)
        combined=torch.cat([aggregated_seq, prev_targets.unsqueeze(-1)], dim=-1)
        # Pass through your LSTM
        return self.lstm_model(combined)

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=20):
    #initialize some variables
    avg_train_losses=[]
    avg_val_losses=[]
    best_val_loss=float('inf')
    best_model_state=None

    for epoch in range(epochs):
        #train loop
        model.train()
        train_loss = 0
        for X, prev_targets, y in train_loader:
            X, prev_targets, y = X.to(device), prev_targets.to(device).float(), y.to(device).float()
            optimizer.zero_grad()
            preds = model(X, prev_targets)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
        
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_losses.append(avg_train_loss)

        #val loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, prev_targets, y in val_loader:
                X, prev_targets, y = X.to(device),prev_targets.to(device).float(), y.to(device).float()
                preds = model(X, prev_targets)
                loss = criterion(preds, y)
                val_loss += loss.item() * X.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        avg_val_losses.append(avg_val_loss)
        if scheduler:
            scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")   

        #save best model
        if avg_val_loss<best_val_loss:
            best_val_loss=avg_val_loss
            best_model_state=deepcopy(model.state_dict())

    #plot the losses
    plt.figure()
    plt.plot(avg_train_losses, label='train losses')
    plt.plot(avg_val_losses, label='val losses')
    plt.xlabel('epochs')
    plt.ylabel('losses')
    plt.legend()

    model.load_state_dict(best_model_state)
    return model


#####################################################################################################################
# Load ALL Features, Aggregate, Create Labels and Folder Mapping 
path_to_mainfolder = "outputs/all_images_encoded"

all_image_folders = sorted(listdir(path_to_mainfolder))
num_folders_total = len(all_image_folders)

df_TSS=pd.read_csv('data/SST_TSS.csv', index_col=0)

features_per_date={}
target_per_date={}
feature_folder_map = [] # Keep track of which folder index each feature belongs to
processed_folder_indices = [] # Keep track of folders we actually found data for
folder_idx_counter = 0 
for folder in all_image_folders:
    if folder not in target_per_date:
        target_per_date[folder]=[]
    # Path to embeddings for this folder
    path_to_folder = f"{path_to_mainfolder}/{folder}"
    images_in_folder_count = 0
    for subfolder in listdir(path_to_folder):
        if subfolder:
            path_to_subfolder=f"{path_to_folder}/{subfolder}"
            if not os_path.exists(path_to_subfolder) or not listdir(path_to_subfolder):
                print(f"Warning: Embeddings path not found or empty for folder {folder}/{subfolder}, skipping.")
                continue
            images_list_embeddings = listdir(path_to_subfolder)
            for image_file in images_list_embeddings:
                if folder not in features_per_date:
                    features_per_date[folder]=[]
                try:
                    # Save features
                    img_path = f"{path_to_subfolder}/{image_file}"
                    embedding = torch.load(img_path).cpu().numpy() # Shape (Channel, Height, Width) e.g. (8, 24, 32)
                    features_per_date[folder].append(embedding)
                    images_in_folder_count += 1
                except Exception as e:
                    print(f"Error loading or processing {img_path}: {e}")
                feature_folder_map.append(folder_idx_counter) # Store the *processed* folder index
    if images_in_folder_count > 0:
        processed_folder_indices.append(folder_idx_counter) # Add original index if processed
        target_per_date[folder]=df_TSS['SST_TSS'].loc[folder].item() # Save label per date
    folder_idx_counter += 1 # Move to the next folder's error value
    
feature_folder_map = np.array(feature_folder_map) 
seq_len=5
#### test dataset class
dataset = DailySequenceDataset(features_per_date, target_per_date, seq_len=seq_len, n_images=32)

# Check how many sequences were created
print(f"Number of sequences: {len(dataset.samples)}")
# Check first sequence shapes and target
for i, (features, prev_targets, target) in enumerate(dataset.samples):
    print(f"Sequence {i} feature shape: {features.shape}")  # should be (seq_len, n_images, feature_dim)
    print(f"Sequence {i} prev targets: {prev_targets}")
    print(f"Sequence {i} target: {target}")

# --- Split Based on Time (Processed Folders) ---
# Use the number of *processed* folders for splitting
train_indices_folders=np.arange(0, 250) 
val_indices_folders = np.arange(250, 300) 

train_set = Subset(dataset, train_indices_folders)
val_set = Subset(dataset, val_indices_folders)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=True)

## define and train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FullPipelineModel(input_dim=1028, 
                          #transformer parameters
                          embed_dim=128,
                          n_heads=8, 
                          n_layers=2,
                          agg_output_dim=128,
                          #lstm parameters
                          lstm_hidden=32,
                          num_layers=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, factor=0.75)
criterion = nn.MSELoss()

trained_model=train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=500)

### testing
full_loader = DataLoader(dataset, batch_size=1, shuffle=False)
all_preds=[]
trained_model.eval()
with torch.no_grad():
    for X, prev_targets, y in full_loader:
        X, prev_targets, y = X.to(device),prev_targets.to(device).float(), y.to(device).float()
        preds = trained_model(X, prev_targets)
        all_preds.append(preds.cpu().item())

all_image_folders_datetime=pd.to_datetime(all_image_folders)
df_TSS.index=pd.to_datetime(df_TSS.index)

# --- Construct Model preds ---
TSS_predictions = pd.Series(
    all_preds,
    index=all_image_folders_datetime[seq_len-1:]
)

# --- Plotting Results ---
plt.figure(figsize=(14, 3), dpi=200)
plt.rcParams.update({'font.size': 12})    
plt.plot(df_TSS['SST_TSS'].iloc[seq_len-1:], '.-', label='Measurements', color='blue')
plt.plot(TSS_predictions.iloc[0:len(train_indices_folders)], '.-', label='predictions (train)', color='orange')
plt.plot(TSS_predictions.iloc[len(train_indices_folders):len(train_indices_folders)+len(val_indices_folders)], '.-', label='predictions (val)', color='green')
plt.plot(TSS_predictions.iloc[len(train_indices_folders)+len(val_indices_folders):], '.-', label='predictions (test)', color='red')
plt.xlabel("Time")
plt.ylabel("TSS (mg/L)")
plt.legend()
plt.show()

