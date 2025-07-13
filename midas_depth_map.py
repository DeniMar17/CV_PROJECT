import os
import glob
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm


#################################################################
########################## CONFIG ###############################
#################################################################
# The following paths are commented or de-commented according to their use and necessity

frame_skip = 1 # Number of frames to skip to compute the depth map

# Paths for real and fake datasets (only one set is currently active)

# real_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/real"
# fake_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/Deepfakes"
#real_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/Face2Face"
#fake_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/FaceSwap"

# real_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/real_actors"
# fake_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/DeepFakeDetection"

real_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test_real/real"
# fake_input_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test/Deepfakes"

#output_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Midas_depth"
output_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Midas_depth_test_real"

# List of directories to process (currently only real)
#all_input_dirs = [real_input_dir, fake_input_dir]
all_input_dirs = [real_input_dir]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Set the computation device (GPU if available, otherwise CPU)
print(f"MiDaS will use device: {device}")

#################################################################
####################### MIDAS SETUP #############################
#################################################################
midas_model = None
midas_transforms = None

# Loads MiDaS depth estimation model and its transforms from torch hub
def load_midas_model_manual():
    global midas_model, midas_transforms
    if midas_model is None:
        print("Loading MiDaS model (DPT_Large)...")
        model_type = "DPT_Large" # Choose MiDaS model type
        midas_model = torch.hub.load("intel-isl/MiDaS", model_type).to(device) # Load the model from torch hub
        midas_model.eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")  # Load the corresponding image transforms
        print("MiDaS model loaded successfully.")
    return midas_model, midas_transforms

# Estimate the depth map of a single image using MiDaS
def estimate_depth_midas(image_path):
    model, transforms = load_midas_model_manual() # Load the image from disk
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR (OpenCV format) to RGB
    input_tensor = transforms.dpt_transform(img).to(device) # Apply MiDaS image preprocessing
    with torch.no_grad():
        # Resize the depth map to match the original image size
        prediction = model(input_tensor)
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()
    return prediction.cpu().numpy()

# Save the depth map as a .npy file
def save_depth_map_npy(depth_map_array, output_path):
    np.save(output_path, depth_map_array)

# Save the depth map as a color image (.png)
def save_depth_map_png(depth_map_array, output_path):
    norm = cv2.normalize(depth_map_array, None, 0, 255, cv2.NORM_MINMAX) # Normalize values between 0–255
    norm = norm.astype(np.uint8)
    color_depth = cv2.applyColorMap(norm, cv2.COLORMAP_MAGMA) # Apply color map for visualization
    cv2.imwrite(output_path, color_depth)


#################################################################
########################## MAIN #################################
#################################################################


os.makedirs(output_dir, exist_ok=True)
print(f"Starting depth map generation\nOutput: {output_dir}\nFrame skip: {frame_skip}\n")

# PHASE 1: Global min/max depth calculation 
print("\nComputing global min/max depth values...")

video_dirs = [] # Collect all subfolders (one per video) from input directories
for input_dir in all_input_dirs:
    video_dirs += sorted([
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
# Initialize min/max depth values
global_min = float('inf')
global_max = float('-inf')

for video_path in tqdm(video_dirs, desc="Scan min/max"): # Loop through all videos and frames to compute global min/max
    frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg"))) # Get all JPG frames in the current video folder
    for i in range(0, len(frame_paths), frame_skip):
        frame_path = frame_paths[i]
        try:
            depth_map = estimate_depth_midas(frame_path)
            global_min = min(global_min, depth_map.min())
            global_max = max(global_max, depth_map.max())
        except:
            continue

if global_max - global_min < 1e-6: # If the range is too small, fallback to 1.0 to avoid division by 0
    print("Depth range too small, falling back to 1.0")
    depth_range = 1.0
else:
    depth_range = global_max - global_min

print(f"global_min: {global_min:.4f}, global_max: {global_max:.4f}, range: {depth_range:.4f}")

# PHASE 2: Normalization and saving 
for video_path in tqdm(video_dirs, desc="Video"):
    # Choose dataset label (real/fake/etc.)

    #dataset_type = "real" if real_input_dir in video_path else "Deepfakes"
    #dataset_type = "Face2Face" if real_input_dir in video_path else "FaceSwap"
    #dataset_type = "real_actors" if real_input_dir in video_path else "DeepFakeDetection"
    dataset_type = "real"  # Adjust based on input
    video_name = os.path.basename(video_path)
    output_video_dir = os.path.join(output_dir, dataset_type, video_name)
    os.makedirs(output_video_dir, exist_ok=True)

    frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg"))) # Get all frame paths

    if not frame_paths:
        tqdm.write(f"No frames found in: {video_name}")
        continue

    for i in range(0, len(frame_paths), frame_skip): # Loop through frames with skipping
        frame_path = frame_paths[i]
        frame_number = f"{i:04d}"
        # Define output paths
        out_npy = os.path.join(output_video_dir, f"depth_{frame_number}.npy")
        out_png = os.path.join(output_video_dir, f"depth_{frame_number}.png")

        if os.path.exists(out_npy):
            continue

        try:
            depth_map = estimate_depth_midas(frame_path) # Estimate and normalize depth
            depth_map_normalized = (depth_map - global_min) / depth_range

            save_depth_map_npy(depth_map_normalized, out_npy) # Save both formats
            save_depth_map_png(depth_map_normalized, out_png)

        except Exception as e:
            tqdm.write(f"Error on frame {frame_number} in {video_name}: {e}")
