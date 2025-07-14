# Truth in Motion: Depth and Flow Enhanced DeepFake Detection
![Image](https://github.com/user-attachments/assets/9353894b-e827-4944-b891-3c0bbcdf8e8e)

Powerful tools and software for creating and processing multimedia content have been made
available by recent advances in visual media technology. In particular, AI-driven methods made the production
of Deepfakes videos simpler than ever. These fake and manioulated videos are dangerous because they can be
used to alter public opinion or harm reputations. Considering these dangers, creating trustworthy techniques
to identify Deepfakes is crucial to preserving data integrity. For this reason, accurately separating fake videos
from authentic ones while maintaining a method that maximizes computational resources is the main goal
of this work.


### Datasets
For this project, the uncompressed videos from FaceForensics++ were used, with three types of deepfake manipulations: Face2Face, Deepfakes, and FaceSwap. Both real and manipulated videos were processed using the following pipeline: face recognition and extraction of centered consecutive frames, optical flow computation using RAFT, and depth estimation using MiDAS, followed by data normalization to facilitate processing. In this Google Drive folder, [CV_PROJECT_DATASET](https://drive.google.com/drive/folders/1CFY5EAeED3pZIpis0zUa_HhKglIrDsmV?usp=drive_link), each video used for dataset creation is represented by 20 frames, along with the corresponding 20 optical flow maps and 20 depth maps.

### Utils
As said before to compute optical flow and depth, RAFT and MiDaS were used, downloaded from their respective GitHub repositories. In this Google Drive folder [CV_PROJECT_UTILS](https://drive.google.com/drive/folders/1CRFdyTP4Y9hI03PMC3qJlPBSmPi4jws1?usp=drive_link), you can find the directories containing the corresponding programs.

### Models
The generated models — specifically, the one produced during training and used for inference, the pruned model, and the model obtained after pruning and quantization — can be found in the following Google Drive folder [CV_PROJECT_MODELS](https://drive.google.com/drive/folders/1DUYvY1-5Mv6dMm_4NY-rWAurp5BeK0UF?usp=drive_link)


> [!IMPORTANT]
> To eventually run the code please check the dependencies file. This project was run on a GEFORCE RTX 5090 using Pytorch and related libraries for Cuda version cu12.9, so maybe you need to install a different Pytorch version for your Cuda version.

### Code in this repository

**_RAFT_of.py_:** uses RAFT model and Insightface to detect and crop faces from the central part of a video and compute optical flow on frames exctracted

**_optical_flow_normalization.py_:** is a two steps algorithm that computes the min/max values in .npy flows and then normalizes all the optical flows extracted

**_midas_depth_map.py_:** uses MiDaS depth estimation model to compute depth of frames and normalizes them using a min/max normalization

**_feature_extractor.py_:** is the main pipeline that takes the preprocessed dataset and a CNN backbone using ResNet18, modified for multimodal input with a Transformer head for temporal aggregation of frame features to extract features and classification

**_feature_extractor_test.py_:** is the test script for inference using the model saved after training

**_pruning_and_quantization.py_:** is a script for prune and quantize the model 


