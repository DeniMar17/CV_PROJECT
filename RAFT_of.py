import sys
sys.path.append('core')

import argparse
import os
import re
import glob
import cv2
import numpy as np
import torch
from PIL import Image

# Import FaceAnalysis class from InsightFace, a face analysis framework
from insightface.app import FaceAnalysis

# Import components from the RAFT optical flow model
from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder

DEVICE = 'cuda'

# Initialize the face analysis application and set the inference to run on CPU only
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640)) # Prepare the face detection model with context ID and target input size

#################################################################
########################## UTILS ################################
#################################################################

# Write optical flow to .flo file 
def write_flo(filename, flow):
    with open(filename, 'wb') as f: # Open the target file in binary write mode
        magic = np.array([202021.25], dtype=np.float32) # Write the magic number that identifies the file as a .flo file (Middlebury format)
        h, w = flow.shape[1], flow.shape[2]  # Get the height and width of the flow field
        magic.tofile(f)
        # Write the width and height as 32-bit integers
        np.array([w], dtype=np.int32).tofile(f)
        np.array([h], dtype=np.int32).tofile(f)
        flow = flow.transpose(1, 2, 0) # Rearrange the flow array from (2, H, W) to (H, W, 2) to match .flo file expectations
        flow.astype(np.float32).tofile(f)  # Convert to float32 and write the flow data to the file

# Resizes an image to fit inside a target resolution while preserving aspect ratio, adding black padding as needed
def letterbox_resize(image, target_size=(224, 224)):
    target_w, target_h = target_size
    h, w = image.shape[:2] # Get original image dimensions
    scale = min(target_w / w, target_h / h) # Calculate scale factor to fit the image inside the target size while preserving aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)  # Compute new image size after scaling
    resized = cv2.resize(image, (new_w, new_h)) # Resize the original image using OpenCV

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8) # Create a blank (black) canvas with the target size
    # Compute top and left margins to center the resized image in the canvas
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized  # Paste the resized image onto the canvas at the computed position
    return canvas

# Detects and crops faces from the central part of a video using InsightFace, applies letterbox resizing, and saves the frames
def extract_faces_insightface(video_path, output_folder, duration=10, target_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get the total number of frames in the video
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # Get the original frame rate (frames per second)

    mid_frame = total_frames // 2  # Determine the middle frame of the video
    half_range = int(duration * original_fps // 2) # Compute the number of frames to process based on desired duration centered around mid_frame
    # Define center portion of the video as range to process
    start_frame = max(0, mid_frame - half_range)
    end_frame = min(total_frames, mid_frame + half_range)

    frame_id = 0
    saved_frame_paths = []

    for i in range(total_frames):
        ret, frame = cap.read() # Read one frame
        if not ret:
            break
        if i < start_frame or i > end_frame:   # Skip frames outside the central range
            continue

        faces = app.get(frame) # Detect faces using InsightFace
        if faces:
            # Use the most confident detected face (first in the list)
            box = faces[0].bbox.astype(int) # Get bounding box and convert to integers
            x1, y1, x2, y2 = box
            face_crop = frame[y1:y2, x1:x2] # Crop the face region from the frame

            face_crop = letterbox_resize(face_crop, target_size)

            out_path = os.path.join(output_folder, f'frame_{frame_id:04d}.jpg')
            cv2.imwrite(out_path, face_crop)
            saved_frame_paths.append(out_path) # Keep track of saved frame path
            frame_id += 1

    cap.release() # Release the video capture object
    return saved_frame_paths

# Loads an image, converts it to a tensor with shape (1, 3, H, W), and moves it to the CUDA device
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8) # Open the image file and convert it to a NumPy array with uint8 values
    img = torch.from_numpy(img).permute(2, 0, 1).float() # Convert the NumPy array to a PyTorch tensor and rearrange dimensions from (H, W, C) to (C, H, W)
    return img[None].to(DEVICE) # Add a batch dimension (1, C, H, W) and move the tensor to the CUDA device

# Pads two input images and computes the dense optical flow using the RAFT model
def compute_flow(image1, image2, model):
    padder = InputPadder(image1.shape) # Initialize the InputPadder to handle padding
    image1, image2 = padder.pad(image1, image2) # Pad both input images to the same size with dimensions divisible by 8
    flow_low, flow_up = model(image1, image2, iters=20, test_mode=True) # Run the RAFT model to compute optical flow, 'test_mode=True' disables gradient computation
    return flow_up[0]  # (2, H, W)

# For a given video, extracts face crops, computes flow between consecutive face frames, and saves the flow as .npy, .flo, and visualization images
def process_video(video_path, output_root, model):
    video_name = os.path.splitext(os.path.basename(video_path))[0] # Extract the video file name without extension
    video_out_folder = os.path.join(output_root, video_name)

    # Skip processing if output folder already exists and is not empty
    if os.path.exists(video_out_folder) and os.listdir(video_out_folder):
        print(f"Skipping {video_name}: output already exists.")
        return

    os.makedirs(video_out_folder, exist_ok=True)
    print(f"Processing video: {video_name}")

    frame_paths = extract_faces_insightface(video_path, video_out_folder) # Extract face frames from the central portion of the video
    # If fewer than 2 face frames were detected, skip optical flow computation
    if len(frame_paths) < 2:
        print(f"Skipping {video_name}, not enough detected face frames.")
        return

    for i in range(len(frame_paths) - 1): # Iterate over consecutive pairs of face frames to compute optical flow
        image1 = load_image(frame_paths[i]) # Load two consecutive face images and prepare them as tensors
        image2 = load_image(frame_paths[i + 1])
        flow_tensor = compute_flow(image1, image2, model) # Compute the dense optical flow between the two images using RAFT
        flow_numpy = flow_tensor.cpu().numpy() # Convert the flow tensor to a NumPy array (on CPU) for saving

        flow_base = os.path.join(video_out_folder, f'flow_{i:04d}')
        np.save(f"{flow_base}.npy", flow_numpy) # Save the flow as a .npy file

        flow_img = flow_viz.flow_to_image(flow_numpy.transpose(1, 2, 0)) # Convert the flow to a color visualization image and save as PNG
        cv2.imwrite(f"{flow_base}_vis.png", flow_img[:, :, [2, 1, 0]]) # Convert RGB to BGR for OpenCV
        write_flo(f"{flow_base}.flo", flow_numpy) # Save the flow in .flo format

        print(f"Saved flow and vis: {flow_base}")


#################################################################
###################### MAIN PIPELINE ############################
#################################################################

# Runs the full processing pipeline: model loading, video filtering, face extraction, and flow computation
def run_optical_flow_pipeline(args):
    os.makedirs(args.output, exist_ok=True)

    model = torch.nn.DataParallel(RAFT(args)) # Initialize the RAFT model using DataParallel to support multi-GPU (if available)
    model.load_state_dict(torch.load(args.model)) # Load the pretrained model weights from file
    model = model.module # Access the actual model inside the DataParallel wrapper
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        video_paths = glob.glob(os.path.join(args.path, '*.mp4')) # Find all .mp4 video files in the input directory
        filtered_video_paths = []

        for vp in video_paths:
            basename = os.path.basename(vp)  # Extract the base filename from the full path
            match = re.search(r'(\d+)', basename) # Extract numeric ID from the filename using regular expressions
            if match:
                video_id = int(match.group(1))

                # This block is commented but can be de-commented if necessary and comment the following one
                # if args.max_id is None or video_id < args.max_id: # Keep the video if it's less than max_id range
                #     filtered_video_paths.append(vp)

                if args.max_id is None or args.min_id < video_id < args.max_id: # Keep the video if it's within the min_id and max_id range
                    filtered_video_paths.append(vp)

        if not filtered_video_paths:
            print("No videos matched the filtering criteria.")
        else:
            print(f"Found {len(filtered_video_paths)} video(s) with ID < {args.max_id}")

        for video_path in filtered_video_paths: # Process each video one by one
            try:
                process_video(video_path, args.output, model)
            except:
                print(f"Exception {video_path}")


#################################################################
########################## MAIN #################################
#################################################################
# The arguments in main are commented or de-commented in base of their use and necessary
# This script is used to produce frames and optical flow for training, validation and test sets
if __name__ == '__main__':
    import argparse

    args = argparse.Namespace(
        model='RAFT/models/raft-things.pth',   # path to the checkpoint
        path='/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/dataset_espanso/original_sequences/youtube/raw/videos/',                         # <-- cartella con i video
        output='/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow_test_real/real',     
        min_id=500,
        max_id=600,                      
        small=False,                         
        mixed_precision=False,
        alternate_corr=False
    )

    run_optical_flow_pipeline(args)

    # args = argparse.Namespace(
    #     model='RAFT/models/raft-things.pth',   # path to the checkpoint
    #     path='/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/dataset_espanso/manipulated_sequences/FaceSwap/raw/videos/',                         # <-- cartella con i video
    #     output='/mnt/d/AIRO/Primo anno/secondo semestre/computer vision/homework/Raft_flow/FaceSwap',         
    #     min_id=192,
    #     max_id=500,                          
    #     small=False,                   
    #     mixed_precision=False,
    #     alternate_corr=False
    # )
    # run_optical_flow_pipeline(args)
