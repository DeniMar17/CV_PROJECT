import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
from PIL import Image
from glob import glob
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


from feature_extractor_2 import DeepfakeDetector, target_size, num_frames, batch_size, device
model_save_path = "best_model2.pth" # Path to the trained model weights


#################################################################
######################### DATASET ###############################
#################################################################
# Custom Dataset class for loading test samples
class DeepfakeTestDataset(Dataset):
    def __init__(self,
                 real_frame_root, real_flow_root, real_depth_root,
                 fake_frame_root, fake_flow_root, fake_depth_root,
                 num_frames=7, target_size=(224, 224),
                 max_fake_per_class=33):
        
        self.samples = []
        self.num_frames = num_frames
        self.target_size = target_size
        self.transforms = T.Compose([ # Image transformation pipeline
            T.Resize(target_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Load all real samples from real test root
        self._load_class_samples(real_frame_root, real_flow_root, real_depth_root, label=0,
                                 max_videos=None, label_dir='real')

        # Load limited fake samples (33 per class) from full dataset
        fake_classes = ['Deepfakes', 'Face2Face', 'FaceSwap']
        for cls in fake_classes:
            self._load_class_samples(fake_frame_root, fake_flow_root, fake_depth_root, label=1,
                                     max_videos=max_fake_per_class, label_dir=cls)

        # Show label distribution for verification
        from collections import Counter
        print("Label distribution:", Counter([s['label'] for s in self.samples]))

    def _load_class_samples(self, frame_root, flow_root, depth_root, label, max_videos, label_dir):
        frame_dir = os.path.join(frame_root, label_dir) # Construct full paths for frames, flow, and depth for the given class
        flow_dir = os.path.join(flow_root, label_dir)
        depth_dir = os.path.join(depth_root, label_dir)

        video_ids = sorted(os.listdir(frame_dir)) # List video folders
        if label == 1: # For fakes, skip first 400 videos (optional filtering)
            video_ids = video_ids[400:]  

        count = 0
        for vid in tqdm(video_ids, desc=f"Loading {label_dir}", leave=False):
            if max_videos is not None and count >= max_videos:
                break
            # Build path to individual video folders
            f_vid_path = os.path.join(frame_dir, vid)
            fl_vid_path = os.path.join(flow_dir, vid)
            d_vid_path = os.path.join(depth_dir, vid)

            if not (os.path.isdir(f_vid_path) and os.path.isdir(fl_vid_path) and os.path.isdir(d_vid_path)):
                continue
            # Gather sorted file paths
            frame_paths = sorted(glob(os.path.join(f_vid_path, "*.jpg")))
            flow_paths = sorted(glob(os.path.join(fl_vid_path, "*.npy")))
            depth_paths = sorted(glob(os.path.join(d_vid_path, "*.npy")))

            min_len = min(len(frame_paths), len(flow_paths), len(depth_paths)) # Only add samples with enough frames
            if min_len >= self.num_frames:
                self.samples.append({
                    'frames': frame_paths[:self.num_frames],
                    'flows': flow_paths[:self.num_frames],
                    'depths': depth_paths[:self.num_frames],
                    'label': label
                })
                count += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames, flows, depths = [], [], []

        for f_path, fl_path, d_path in zip(sample['frames'], sample['flows'], sample['depths']): # Load and preprocess each modality
            img = Image.open(f_path).convert("RGB")
            frames.append(self.transforms(img))

            flow = torch.tensor(np.load(fl_path), dtype=torch.float32)
            if flow.ndim == 2:
                flow = flow.unsqueeze(0) # Shape: (1, H, W)
            elif flow.ndim == 3 and flow.shape[0] > 2:
                flow = flow[:2] # Only keep first two channels
            flow = torch.nn.functional.interpolate(flow.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            flows.append(flow)

            depth = torch.tensor(np.load(d_path), dtype=torch.float32)
            if depth.ndim == 2:
                depth = depth.unsqueeze(0)
            elif depth.ndim == 3 and depth.shape[0] > 1:
                depth = depth[:1]
            depth = torch.nn.functional.interpolate(depth.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
            depths.append(depth)

        return torch.stack(frames), torch.stack(flows), torch.stack(depths), torch.tensor(sample['label'], dtype=torch.float32)


#################################################################
###################### MODEL SETUP ##############################
#################################################################
model = DeepfakeDetector()
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode

print("Loading test set...") # Load the test dataset
test_dataset = DeepfakeTestDataset(
    real_frame_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test_real",
    real_flow_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_normalized_test_real",
    real_depth_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Midas_depth_test_real",

    fake_frame_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow",
    fake_flow_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_normalized",
    fake_depth_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Midas_depth" ,

    num_frames=num_frames,
    target_size=target_size,
    max_fake_per_class=33
)
# Create DataLoader for batching during testing
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#################################################################
######################## TEST LOOP ##############################
#################################################################
print("Starting test...")
all_preds, all_labels = [], []

with torch.no_grad(): # Disable gradient tracking for evaluation
    for rgb, flow, depth, labels in tqdm(test_loader, desc="Testing"):
        rgb, flow, depth, labels = rgb.to(device), flow.to(device), depth.to(device), labels.to(device) # Move data to device (GPU/CPU)
        outputs = model(rgb, flow, depth).squeeze(1) # Get model predictions
        preds = (torch.sigmoid(outputs) > 0.5).float() # Convert logits to binary predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#################################################################
########################## METRICS ##############################
#################################################################
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["Real", "Fake"]))

#################################################################
###################### CONFUSION MATRIX #########################
#################################################################
cm = confusion_matrix(all_labels, all_preds)
labels = ["Real", "Fake"]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
output_path = "confusion_matrix.png"
plt.savefig(output_path)
print(f"Confusion saved in: {output_path}")
plt.close()
