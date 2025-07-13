import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import copy
from feature_extractor_2 import DeepfakeDetector, DeepfakeDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.ao.quantization import quantize_dynamic
from torch.serialization import add_safe_globals
from feature_extractor_2 import DeepfakeDetector

add_safe_globals([DeepfakeDetector]) # Add custom model class to torch's serialization registry


#################################################################
########################## CONFIG ###############################
#################################################################
original_model_path = "best_model2.pth" # Pretrained model path
pruned_model_path = "pruned_model.pth" # Path to save pruned model
quantized_model_path = "pruned_and_quantized_model.pt" # Path to save final quantized model
device = torch.device("cpu")
batch_size = 4
fine_tune_epochs = 3 # Number of fine-tuning epochs after pruning
learning_rate = 1e-5 # Low LR for gentle fine-tuning

print("Loading original model")
model = DeepfakeDetector()
model.load_state_dict(torch.load(original_model_path, map_location=device))
model.eval()

# Clone model to avoid modifying the original
pruned_model = copy.deepcopy(model)

# Apply 30% L1 unstructured pruning to Conv2d and Linear layers
print("Applying 30% unstructured pruning to Conv2d and Linear layers...")
for name, module in pruned_model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name="weight", amount=0.3)
        prune.remove(module, "weight")

#################################################################
########################## DATASET ##############################
#################################################################
print("Loading dataset for fine-tuning...")
full_dataset = DeepfakeDataset(
    frame_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow",
    flow_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_normalized",
    depth_root="/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Midas_depth"
)
# Use 50% of data for fine-tuning
train_idx, _ = train_test_split(list(range(len(full_dataset))), test_size=0.5, random_state=42)
train_loader = DataLoader(torch.utils.data.Subset(full_dataset, train_idx), batch_size=batch_size, shuffle=True)

# Fine-tuning the pruned model
print(f"Starting fine-tuning for {fine_tune_epochs} epochs...")
pruned_model.train()
pruned_model.to(device)
# Set up optimizer and loss
optimizer = torch.optim.Adam(pruned_model.parameters(), lr=learning_rate)
loss_fn = nn.BCEWithLogitsLoss() # Binary classification loss (expects logits)

for epoch in range(fine_tune_epochs):
    pruned_model.train()
    total_loss = 0.0
    num_batches = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{fine_tune_epochs}", leave=False)

    for rgb, flow, depth, labels in loop:
        rgb, flow, depth, labels = rgb.to(device), flow.to(device), depth.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = pruned_model(rgb, flow, depth).squeeze(1)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Epoch {epoch+1}/{fine_tune_epochs} - Avg Loss: {avg_loss:.4f}")

pruned_model.eval()

# Save the pruned and fine-tuned model
torch.save(pruned_model.state_dict(), pruned_model_path)
print(f"Pruned model saved at: {pruned_model_path}")

# Apply dynamic quantization to the classifier head
print("Applying dynamic quantization to classifier layer...")
pruned_model.transformer.classifier = quantize_dynamic(
    pruned_model.transformer.classifier,
    {nn.Linear},
    dtype=torch.qint8
)

torch.save(pruned_model, quantized_model_path)

print(f"Quantized model saved at: {quantized_model_path}")

# Load quantized model for testing
quantized_model = torch.load(quantized_model_path, map_location=device, weights_only=False)
quantized_model.eval()

# Evaluate quantized model on 10 samples
print("\nTesting quantized model on 10 samples:")
test_loader = DataLoader(full_dataset, batch_size=1, shuffle=True)

with torch.no_grad(): # Run inference on a few samples
    count = 0
    for rgb, flow, depth, labels in test_loader:
        rgb, flow, depth, labels = rgb.to(device), flow.to(device), depth.to(device), labels.to(device)
        outputs = quantized_model(rgb, flow, depth).squeeze(1)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        print(f"[{count+1}] Prediction: {preds.item()} | True label: {labels.item()}")
        count += 1
        if count == 10:
            break

