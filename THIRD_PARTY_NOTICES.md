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

- File: `face_landmarks_detector.onnx`
- ONNX conversion: https://huggingface.co/FreeHugsForRobots/ps-face-landmarks
- Original model project: https://github.com/google-ai-edge/mediapipe
- License declared by the model distributor: Apache License 2.0
- SHA-256: `9c8dbae0cffd7b8e195b7c5e3795bd2a0f206a06b27edf30b2dd6900175c652a`

The ONNX distributor identifies the file as a format conversion of Google's MediaPipe Face Mesh /
Face Landmarker model with no model modifications. MediaPipe's repository and Face Mesh V2 model
card carry the Apache License 2.0. Keep this notice and the Apache License 2.0 text with commercial
distributions that include the model.
