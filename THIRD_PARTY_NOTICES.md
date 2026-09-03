# Third-party face preprocessing models

LatentSync downloads these pinned model files into `checkpoints/auxiliary`. The checkpoint
directory is excluded from source control.

## YuNet face detector

- Source: OpenCV Zoo, `face_detection_yunet_2026may.onnx`
- Upstream: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- License: MIT
- SHA-256: `ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0`

Copyright notices and the full license text are available in the upstream model directory.

## MediaPipe Face Landmarker v2 (478 points)

- File: `face_landmarker_Nx3x256x256.onnx`
- ONNX/PyTorch port and preprocessing implementation:
  https://github.com/yakhyo/mediapipe-face-mesh-onnx
- Original model project: https://github.com/google-ai-edge/mediapipe
- License: Apache License 2.0
- SHA-256: `111795f8703cdeb6d0c68a9f3cc966a0f23f8786bb00f4577a11f461fc4276ac`

The model architecture, ROI transform, preprocessing, and coordinate mapping are adapted from
Yakhyokhuja Valikhujaev's Apache-2.0 implementation. Its weights originate from Google's MediaPipe
Face Landmarker model. Keep this notice and the Apache License 2.0 text with commercial
distributions that include the model or adapted implementation.
