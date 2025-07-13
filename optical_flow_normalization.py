import os
import numpy as np
import glob
from tqdm import tqdm

#################################################################
########################## CONFIG ###############################
#################################################################
# The following paths are commented or de-commented according to their use and necessity


# real_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/real"
# fake_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/Deepfakes"

# real_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/Face2Face"
# fake_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/FaceSwap"

# real_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/real_actors"
# fake_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/DeepFakeDetection"

real_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test_real/real"
output_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_normalized_test_real"

# real_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test/real"
# fake_flow_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test/Deepfakes"
# output_dir = "/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test_normalized"

epsilon = 1e-6  # Value to avoid zero division

#################################################################
################### STEPS TO NORMALIZATION ######################
#################################################################
# Define which directories to process (real only, or real + fake)

#all_input_dirs = [real_flow_dir, fake_flow_dir]
all_input_dirs = [real_flow_dir]
flow_paths = []

for dir_path in  tqdm(all_input_dirs, desc="Scan paths"): # Recursively find all .npy files in the input directories
    flow_paths.extend(glob.glob(os.path.join(dir_path, "**/*.npy"), recursive=True))

if not flow_paths:
    raise FileNotFoundError("No .npy files found in the specified directories.")

# PHASE 1: Compute global min/max values
print("Computing global min/max across all files...")

global_min = float('inf') # Initialize global min and max
global_max = float('-inf')
# Loop over all flow files to find global min and max values
for path in tqdm(flow_paths, desc="Scan min/max"):
    flow = np.load(path)
    global_min = min(global_min, flow.min())
    global_max = max(global_max, flow.max())


print(f"Global minimum: {global_min}")
print(f"Global maximum: {global_max}")

# PHASE 2: Normalize and save 
print("\nStarting normalization of .npy files...")

for path in tqdm(flow_paths, desc="Video"):
    flow = np.load(path) # Load the optical flow array
    flow_norm = (flow - global_min) / (global_max - global_min + epsilon) # Normalize flow using global min/max

    rel_path = os.path.relpath(path, real_flow_dir) # Build relative path to preserve folder structure
    dest_subdir = "real" # Define output subdirectory (e.g., "real", "fake")

    # You can uncomment and adjust this section if mixing real and fake:
    # if path.startswith(real_flow_dir):
    #     rel_path = os.path.relpath(path, real_flow_dir)
    #     #dest_subdir = "real"
    #     #dest_subdir = "real_actors"
    #     dest_subdir = "Face2Face"
    # else:
    #     rel_path = os.path.relpath(path, fake_flow_dir)
    #     #dest_subdir = "Deepfakes"
    #     #dest_subdir = "DeepFakeDetection"
    #     dest_subdir = "FaceSwap"

    out_path = os.path.join(output_dir, dest_subdir, rel_path) # Create the full output path and save the normalized file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, flow_norm)


print(f"\nNormalization completed. Files saved in: {output_dir}")
