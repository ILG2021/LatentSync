import hashlib
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import torch

from .onnx_face_models import OpenCvYuNet, OrtFaceLandmark, OrtYuNet


YUNET_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2026may.onnx"
)
YUNET_MODEL_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
LANDMARK_MODEL_URL = (
    "https://huggingface.co/FreeHugsForRobots/ps-face-landmarks/resolve/main/"
    "face_landmarks_detector.onnx"
)
LANDMARK_MODEL_SHA256 = "9c8dbae0cffd7b8e195b7c5e3795bd2a0f206a06b27edf30b2dd6900175c652a"


def _ensure_model(model_path: Path, model_url: str, expected_sha256: str) -> Path:
    """Download a pinned model when it is not already installed."""
    if model_path.is_file():
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(
                f"Model checksum mismatch for {model_path}: expected "
                f"{expected_sha256}, got {digest}"
            )
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f"{model_path.name}.",
            suffix=".download",
            dir=model_path.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            request = Request(model_url, headers={"User-Agent": "LatentSync"})
            with urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temporary_file)

        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise RuntimeError(
                f"Downloaded model failed checksum validation: expected "
                f"{expected_sha256}, got {digest}"
            )

        # os.replace is atomic when source and destination are on the same volume.
        os.replace(temporary_path, model_path)
        temporary_path = None
        return model_path
    except (HTTPError, URLError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f"Model is missing and could not be downloaded "
            f"to {model_path}. Run setup_env.ps1 (Windows) or setup_env.sh "
            f"(Linux/macOS), or download it from {model_url}."
        ) from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class FaceDetector:
    def __init__(self, device="cuda", detection_size=None, detection_interval=None):
        project_root = Path(__file__).resolve().parents[2]
        yunet_model_path = project_root / "checkpoints" / "auxiliary" / "face_detection_yunet_2026may.onnx"
        yunet_model_path = _ensure_model(yunet_model_path, YUNET_MODEL_URL, YUNET_MODEL_SHA256)
        if detection_size is None:
            detection_size = int(os.environ.get("LATENTSYNC_FACE_DETECTION_SIZE", "640"))
        if detection_interval is None:
            detection_interval = int(os.environ.get("LATENTSYNC_FACE_DETECTION_INTERVAL", "5"))
        if detection_interval < 1:
            raise ValueError("Face detection interval must be at least 1")
        self.detection_size = detection_size
        self.detection_interval = detection_interval
        self.backend = "cpu"
        self.landmark_backend = "mediapipe-cpu"
        device_id = cuda_to_int(device) if str(device).startswith("cuda") else None
        try:
            if device_id is not None:
                self.detector = OrtYuNet(
                    yunet_model_path, device_id=device_id, max_input_size=detection_size
                )
                self.backend = "onnxruntime-cuda"
            else:
                self.detector = OpenCvYuNet(
                    yunet_model_path, max_input_size=detection_size
                )
        except (ImportError, RuntimeError) as error:
            warnings.warn(
                f"YuNet is using OpenCV CPU because its CUDA backend could not start: {error}",
                RuntimeWarning,
            )
            self.detector = OpenCvYuNet(
                yunet_model_path, max_input_size=detection_size
            )
        # Keep an OpenCV YuNet instance for the first frame so initialization
        # uses the same CPU detector + MediaPipe combination as the stable path.
        self.initialization_detector = (
            self.detector
            if self.backend == "cpu"
            else OpenCvYuNet(yunet_model_path, max_input_size=detection_size)
        )
        self.previous_bbox = None
        self.previous_landmarks = None
        self.frame_index = 0
        self.mesh = None
        self.crop_mesh = None
        self.ort_landmark = None
        landmark_model_path = project_root / "checkpoints" / "auxiliary" / "face_landmarks_detector.onnx"
        if device_id is not None:
            try:
                landmark_model_path = _ensure_model(
                    landmark_model_path, LANDMARK_MODEL_URL, LANDMARK_MODEL_SHA256
                )
                self.ort_landmark = OrtFaceLandmark(landmark_model_path, device_id=device_id)
                self.landmark_backend = "onnxruntime-cuda"
            except (ImportError, RuntimeError, ValueError) as error:
                warnings.warn(
                    f"Face landmarks are using MediaPipe CPU because the CUDA ONNX model "
                    f"could not start: {error}",
                    RuntimeWarning,
                )
        print(
            "Face preprocessing backends: "
            f"YuNet={self.backend}, landmarks={self.landmark_backend}, "
            f"detection_long_edge={self.detection_size}, "
            f"detection_interval={self.detection_interval}"
        )

    def reset_tracking(self):
        """Discard temporal state before processing a new, unrelated clip."""
        self.previous_bbox = None
        self.previous_landmarks = None
        self.frame_index = 0

    def _ensure_mediapipe(self):
        if self.mesh is not None:
            return
        import mediapipe as mp

        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        self.crop_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.2,
        )

    def _detect_landmarks_in_crop(self, frame, detection, force_mediapipe=False):
        f_h, f_w, _ = frame.shape
        if self.ort_landmark is not None and not force_mediapipe:
            return self.ort_landmark.process(frame, detection)

        self._ensure_mediapipe()

        x, y, w, h = detection[:4]
        padding = 0.5
        x1 = max(0, int(np.floor(x - w * padding)))
        y1 = max(0, int(np.floor(y - h * padding)))
        x2 = min(f_w, int(np.ceil(x + w * (1.0 + padding))))
        y2 = min(f_h, int(np.ceil(y + h * (1.0 + padding))))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop_h, crop_w = crop.shape[:2]
        scale = max(1.0, 512.0 / max(crop_h, crop_w))
        if scale > 1.0:
            crop_for_mesh = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        else:
            crop_for_mesh = crop

        result = self.crop_mesh.process(crop_for_mesh)
        if not result.multi_face_landmarks:
            # Preserve the GPU fallback if MediaPipe cannot initialize the
            # landmark track for an unusual first-frame pose.
            if force_mediapipe and self.ort_landmark is not None:
                return self.ort_landmark.process(frame, detection)
            return None
        candidate = result.multi_face_landmarks[0]
        return np.array(
            [[x1 + p.x * crop_w, y1 + p.y * crop_h] for p in candidate.landmark],
            dtype=np.float32,
        )

    @staticmethod
    def _tracking_detection(landmarks, frame_width, frame_height):
        """Build the next ROI using MediaPipe's landmark-to-ROI convention."""
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (478, 2) or not np.isfinite(landmarks).all():
            return None

        lower = landmarks.min(axis=0)
        upper = landmarks.max(axis=0)
        width, height = upper - lower
        if width < 8 or height < 8:
            return None

        # MediaPipe derives tracking rotation from eye-corner landmarks 33 ->
        # 263. The landmark backend applies the 1.5 square ROI expansion.
        x1 = max(0.0, float(lower[0]))
        y1 = max(0.0, float(lower[1]))
        x2 = min(float(frame_width), float(upper[0]))
        y2 = min(float(frame_height), float(upper[1]))
        if x2 <= x1 or y2 <= y1:
            return None

        eye_0 = landmarks[33]
        eye_1 = landmarks[263]
        return np.asarray(
            [x1, y1, x2 - x1, y2 - y1, *eye_0, *eye_1], dtype=np.float32
        )

    def _accept_landmarks(self, landmarks, frame_width, frame_height):
        if landmarks is None:
            return None
        landmarks = np.asarray(landmarks, dtype=np.float32)
        if landmarks.shape != (478, 2) or not np.isfinite(landmarks).all():
            return None

        x1, y1 = np.floor(landmarks.min(axis=0)).astype(int)
        x2, y2 = np.ceil(landmarks.max(axis=0)).astype(int)
        width, height = x2 - x1, y2 - y1
        if width <= 0 or height <= 0:
            return None

        self.previous_landmarks = landmarks
        self.previous_bbox = (x1, y1, x2, y2)
        x1 = max(0, x1 - int(width * 0.05))
        y1 = max(0, y1 - int(height * 0.05))
        x2 = min(frame_width, x2 + int(width * 0.05))
        y2 = min(frame_height, y2 + int(height * 0.10))
        return (x1, y1, x2, y2), np.round(landmarks).astype(np.int32)

    def _run_yunet(self, frame, frame_width, frame_height, score_threshold=None):
        initialize_track = self.previous_landmarks is None and self.ort_landmark is not None
        detector = self.initialization_detector if initialize_track else self.detector
        if initialize_track:
            print("Initializing face track on CPU: YuNet=opencv, landmarks=mediapipe.")
        _, detections = detector.detect(frame, score_threshold=score_threshold)
        if detections is None:
            return None

        detections = list(detections)
        if self.previous_bbox is not None:
            px1, py1, px2, py2 = self.previous_bbox

            def intersection_over_union(item):
                x1, y1, width, height = item[:4]
                x2, y2 = x1 + width, y1 + height
                intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(
                    0.0, min(y2, py2) - max(y1, py1)
                )
                union = width * height + (px2 - px1) * (py2 - py1) - intersection
                return intersection / max(union, 1.0)

            best_iou = max(intersection_over_union(item) for item in detections)
            if best_iou > 0.05:
                detections.sort(key=intersection_over_union, reverse=True)
            else:
                detections.sort(key=lambda item: item[2] * item[3], reverse=True)
        else:
            detections.sort(key=lambda item: item[2] * item[3], reverse=True)

        # Establish a reliable track with the original MediaPipe path. Once
        # landmarks exist, all later frames use the CUDA ONNX landmark model.
        for detection in detections:
            accepted = self._accept_landmarks(
                self._detect_landmarks_in_crop(
                    frame, detection, force_mediapipe=initialize_track
                ),
                frame_width,
                frame_height,
            )
            if accepted is not None:
                return accepted
        return None

    def __call__(self, frame, threshold=None):
        if threshold is not None:
            threshold = float(threshold)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("Face detection threshold must be between 0 and 1")
        f_h, f_w, _ = frame.shape
        # LatentSync's video readers already return RGB frames.
        frame = np.asarray(frame).astype(np.uint8)

        run_yunet = self.previous_landmarks is None or self.frame_index % self.detection_interval == 0
        self.frame_index += 1

        if not run_yunet:
            tracking_roi = self._tracking_detection(self.previous_landmarks, f_w, f_h)
            if tracking_roi is not None:
                accepted = self._accept_landmarks(
                    self._detect_landmarks_in_crop(frame, tracking_roi), f_w, f_h
                )
                if accepted is not None:
                    return accepted

        # Run YuNet on the first frame, periodically, and immediately whenever
        # landmark tracking loses confidence.
        accepted = self._run_yunet(frame, f_w, f_h, score_threshold=threshold)
        if accepted is not None:
            return accepted

        # A periodic detector miss does not invalidate a still-healthy track.
        if run_yunet and self.previous_landmarks is not None:
            tracking_roi = self._tracking_detection(self.previous_landmarks, f_w, f_h)
            if tracking_roi is not None:
                accepted = self._accept_landmarks(
                    self._detect_landmarks_in_crop(frame, tracking_roi), f_w, f_h
                )
                if accepted is not None:
                    return accepted

        # Keep a bounded full-frame path for unusual cases that YuNet does not
        # propose. Processing 4K here (and formerly retrying at 8K) wastes both
        # memory and time. Normalized landmarks map back to the source directly
        # after an aspect-preserving resize.
        self._ensure_mediapipe()
        fallback_scale = min(1.0, self.detection_size / max(f_w, f_h))
        if fallback_scale < 1.0:
            fallback_frame = cv2.resize(
                frame,
                (
                    max(1, int(round(f_w * fallback_scale))),
                    max(1, int(round(f_h * fallback_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        else:
            fallback_frame = frame
        result = self.mesh.process(fallback_frame)
        if not result.multi_face_landmarks:
            return None, None
        best = None
        best_area = 0
        for candidate in result.multi_face_landmarks:
            lmk = np.array([[p.x * f_w, p.y * f_h] for p in candidate.landmark])
            x1, y1 = np.floor(lmk.min(axis=0)).astype(int)
            x2, y2 = np.ceil(lmk.max(axis=0)).astype(int)
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > best_area:
                best, best_area = (lmk, (x1, y1, x2, y2)), area
        if best is None:
            return None, None
        lmk, _ = best
        accepted = self._accept_landmarks(lmk, f_w, f_h)
        return accepted if accepted is not None else (None, None)

    def close(self):
        if self.mesh is not None:
            self.mesh.close()
            self.mesh = None
        if self.crop_mesh is not None:
            self.crop_mesh.close()
            self.crop_mesh = None

        # ONNX Runtime and OpenCV do not expose a close method for these
        # objects. Dropping the final references destroys their sessions/models
        # and releases their CUDA/CPU allocations before diffusion starts.
        self.ort_landmark = None
        self.initialization_detector = None
        self.detector = None
        self.reset_tracking()


def cuda_to_int(cuda_str: str) -> int:
    """
    Convert the string with format "cuda:X" to integer X.
    """
    if cuda_str == "cuda":
        return 0
    device = torch.device(cuda_str)
    if device.type != "cuda":
        raise ValueError(f"Device type must be 'cuda', got: {device.type}")
    return device.index
