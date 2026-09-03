#!/bin/bash

# Install ffmpeg
sudo apt -y install ffmpeg

# Python dependencies
pip install -r requirements.txt

# OpenCV dependencies
sudo apt -y install libgl1

# Download the checkpoints required for inference from HuggingFace
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

# Download the MIT-licensed YuNet face detector from the official OpenCV Zoo.
mkdir -p checkpoints/auxiliary
curl -L \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx \
  -o checkpoints/auxiliary/face_detection_yunet_2026may.onnx

# Download yakhyo's Apache-2.0 MediaPipe Face Landmarker v2 ONNX port (478 points).
curl -L \
  https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights/face_landmarker_Nx3x256x256.onnx \
  -o checkpoints/auxiliary/face_landmarker_Nx3x256x256.onnx
