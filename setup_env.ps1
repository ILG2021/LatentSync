# Install ffmpeg (Using winget for Windows)
# winget install ffmpeg

# Python dependencies
pip install -r requirements.txt

# Download the checkpoints required for inference from HuggingFace
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints

# Download the MIT-licensed YuNet face detector from the official OpenCV Zoo.
New-Item -ItemType Directory -Force checkpoints/auxiliary | Out-Null
Invoke-WebRequest `
    -Uri "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx" `
    -OutFile "checkpoints/auxiliary/face_detection_yunet_2023mar.onnx"
