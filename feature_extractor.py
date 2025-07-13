import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from glob import glob
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
from tqdm import tqdm


#################################################################
########################## CONFIG ###############################
#################################################################
frame_root_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow"
flow_root_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_normalized"
depth_root_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Midas_depth" 
batch_size = 4
num_frames = 7 # Number of frames per sample
target_size = (224, 224) # Resizing resolution
epochs = 15
learning_rate = 1e-4
warmup_epochs = 2
model_save_path = "best_model2.pth" # Path where best model is saved
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#################################################################
######################### DATASET ###############################
#################################################################
class DeepfakeDataset(Dataset):
    def __init__(self, frame_root, flow_root, depth_root, num_frames=7, target_size=(224, 224)):
        self.samples = []
        self.num_frames = num_frames
        self.target_size = target_size
        self.transforms = T.Compose([ # Define image transformations for RGB frames
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        real_class = 'real'
        fake_classes = ['Deepfakes', 'Face2Face', 'FaceSwap']

        # Load up to 500 real videos
        self._load_class_samples(real_class, frame_root, flow_root, depth_root, label=0, max_videos=500)

        # Load up to ~166 videos per fake class
        max_fake_total = 500
        per_class_quota = max_fake_total // len(fake_classes)
        for cls in fake_classes:
            self._load_class_samples(cls, frame_root, flow_root, depth_root, label=1, max_videos=per_class_quota)

    def _load_class_samples(self, label_dir, frame_root, flow_root, depth_root, label, max_videos):
        frame_dir = os.path.join(frame_root, label_dir) # Construct full paths to frame, flow, and depth folders
        flow_dir = os.path.join(flow_root, label_dir)
        depth_dir = os.path.join(depth_root, label_dir)

        video_ids = sorted(os.listdir(frame_dir))[:max_videos]
        for vid in tqdm(video_ids, desc=f"Loading {label_dir}"):
            f_vid_path = os.path.join(frame_dir, vid)
            fl_vid_path = os.path.join(flow_dir, vid)
            d_vid_path = os.path.join(depth_dir, vid)

            if not (os.path.isdir(f_vid_path) and os.path.isdir(fl_vid_path) and os.path.isdir(d_vid_path)):
                continue
            # Gather file paths
            frame_paths = sorted(glob(os.path.join(f_vid_path, "*.jpg")))
            flow_paths = sorted(glob(os.path.join(fl_vid_path, "*.npy")))
            depth_paths = sorted(glob(os.path.join(d_vid_path, "*.npy")))

            min_len = min(len(frame_paths), len(flow_paths), len(depth_paths)) # Ensure enough frames exist for this video
            if min_len >= self.num_frames:
                self.samples.append({
                    'frames': frame_paths[:self.num_frames],
                    'flows': flow_paths[:self.num_frames],
                    'depths': depth_paths[:self.num_frames],
                    'label': label
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames, flows, depths = [], [], []

        for f_path, fl_path, d_path in zip(sample['frames'], sample['flows'], sample['depths']):
            img = Image.open(f_path).convert("RGB") # Load and preprocess RGB frame
            frames.append(self.transforms(img))

            flow = torch.tensor(np.load(fl_path), dtype=torch.float32) # Load and preprocess optical flow
            if flow.ndim == 2:
                flow = flow.unsqueeze(0) # (H, W) -> (1, H, W)
            elif flow.ndim == 3 and flow.shape[0] > 2:
                flow = flow[:2]
            flow = torch.nn.functional.interpolate(flow.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            flows.append(flow)


            depth = torch.tensor(np.load(d_path), dtype=torch.float32) # Load and preprocess depth
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)  # (H, W) → (1, H, W)
            elif depth.ndim == 3 and depth.shape[0] > 1:
                depth = depth[:1]  # Take only first channel
            # Resizing: (1, H, W) → (1, target_H, target_W)
            depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            depths.append(depth)


        rgb_tensor = torch.stack(frames)   # (N, 3, H, W)
        flow_tensor = torch.stack(flows)   # (N, 2, H, W)
        depth_tensor = torch.stack(depths) # (N, 1, H, W)
        label = torch.tensor(sample['label'], dtype=torch.float32)
        return rgb_tensor, flow_tensor, depth_tensor, label


#################################################################
###################### MODEL SETUP ##############################
#################################################################

# CNN backbone using ResNet18, modified for multimodal input
class ResNet18FeatureExtractor(nn.Module):
    def __init__(self, in_channels=6): # 3 (RGB) + 2 (Flow) + 1 (Depth)
        super().__init__()
        base_model = models.resnet18(pretrained=True)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        )
        self.encoder = nn.Sequential(
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.encoder(x)
        return x  # Output: (B, 512, h, w)

# Transformer head for temporal aggregation of frame features
class TransformerHead(nn.Module):
    def __init__(self, input_dim=512, num_heads=8, num_layers=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(input_dim, 1)

    def forward(self, x):  # Input: (B, N, 512)
        x = self.transformer(x)
        return self.classifier(x[:, 0, :]) # Output: (B, 1)

# Full Deepfake Detector: CNN + Transformer
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = ResNet18FeatureExtractor(in_channels=6)
        self.transformer = TransformerHead(input_dim=512)

    def forward(self, rgb_seq, flow_seq, depth_seq):
        B, N, _, H, W = rgb_seq.shape
        features = []
        for i in range(N):
            rgb = rgb_seq[:, i]
            flow = flow_seq[:, i]
            depth = depth_seq[:, i]
            x = torch.cat([rgb, flow, depth], dim=1) # Concatenate all modalities
            feat = self.cnn(x)
            feat = feat.mean(dim=[2, 3]) # Global average pooling
            features.append(feat)

        seq_feat = torch.stack(features, dim=1)  # (B, N, 512)
        return self.transformer(seq_feat)  # (B, 1)

#################################################################
####################### TRAIN LOOP ##############################
#################################################################
def train(model, train_loader, val_loader, patience=5):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # Initialize the Adam optimizer with the model's parameters
    loss_fn = nn.BCEWithLogitsLoss() # Binary classification loss with logits
    # Linear warmup scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: min((epoch + 1) / warmup_epochs, 1.0)
    )
    # Initialize tracking variables
    best_f1 = 0
    best_epoch = 0

    train_losses = []
    val_accuracies = []
    val_f1_scores = []

    for epoch in range(epochs):
        model.train() # Set model to training mode
        total_loss, correct, total = 0, 0, 0
        for rgb, flow, depth, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", leave=False):
            rgb, flow, depth, labels = rgb.to(device), flow.to(device), depth.to(device), labels.to(device) # Move data to the appropriate device
            optimizer.zero_grad() # Reset optimizer gradients
            outputs = model(rgb, flow, depth).squeeze(1) # Forward pass through the model, (B,) after squeezing
            loss = loss_fn(outputs, labels) # Compute binary classification loss
            loss.backward() # Backpropagate gradients
            optimizer.step() # Update model parameters
            # Track loss and accuracy
            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step() # Update learning rate using the scheduler

        acc = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Train Acc: {acc:.2f}%")

        val_acc, val_f1 = validate(model, val_loader)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)

         # Early stopping based on F1
        if val_f1 > best_f1:
            best_f1 = val_f1 # New best F1 score — save model
            best_epoch = epoch
            torch.save(model.state_dict(), model_save_path)
            print(f"  Model saved at epoch {epoch+1} (F1: {val_f1:.4f})")
        elif epoch - best_epoch >= patience: # Stop training if no improvement for `patience` epochs
            print(f"\nEarly stopping at epoch {epoch+1} (no improvement in F1 for {patience} epochs)")
            break


#################################################################
####################### SAVE PLOTS #############################
#################################################################
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Acc')
    plt.plot(val_f1_scores, label='Val F1')
    plt.legend()
    plt.title("Validation Metrics")

    plt.tight_layout()
    plt.savefig("plots/training_metrics.png")
    plt.close()

#################################################################
#################### VALIDATION LOOP ############################
#################################################################
def validate(model, val_loader):
    model.eval() # Set the model to evaluation mode
    correct, total = 0, 0
    all_preds = [] # Lists to collect all predictions and true labels
    all_labels = []
    with torch.no_grad(): # Disable gradient tracking
        for rgb, flow, depth, labels in tqdm(val_loader, desc="Validation", leave=False):
            rgb, flow, depth, labels = rgb.to(device), flow.to(device), depth.to(device), labels.to(device) # Move data to the same device as the model
            outputs = model(rgb, flow, depth).squeeze(1) # Forward pass through the model
            preds = (torch.sigmoid(outputs) > 0.5).float() # Apply sigmoid to convert logits to probabilities, then threshold at 0.5
            correct += (preds == labels).sum().item() # Count number of correct predictions
            total += labels.size(0) # Total number of samples
            all_preds.extend(preds.cpu().numpy()) # Store predictions and labels for F1 and classification report
            all_labels.extend(labels.cpu().numpy())

    acc = 100 * correct / total
    f1 = f1_score(all_labels, all_preds)
    print(f"  Validation Accuracy: {acc:.2f}% - F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'Fake']))
    print()
    
    return acc, f1

#################################################################
########################## MAIN #################################
#################################################################
if __name__ == "__main__":
    dataset = DeepfakeDataset( # Load dataset and split into training and validation sets
        frame_root=frame_root_dir,
        flow_root=flow_root_dir,
        depth_root=depth_root_dir,
        num_frames=num_frames
    )

    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    # Initialize and train the model
    model = DeepfakeDetector()
    train(model, train_loader, val_loader)
