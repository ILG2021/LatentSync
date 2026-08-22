import mediapipe as mp
import cv2
import numpy as np
import torch
from pathlib import Path

class FaceDetector:
    def __init__(self, device="cuda"):
        project_root = Path(__file__).resolve().parents[2]
        yunet_model_path = project_root / "checkpoints" / "auxiliary" / "face_detection_yunet_2023mar.onnx"
        if not yunet_model_path.is_file():
            raise FileNotFoundError(
                f"YuNet face detector model not found: {yunet_model_path}. "
                "Run setup_env.sh or download the MIT-licensed model from OpenCV Zoo."
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(yunet_model_path),
            "",
            (320, 320),
            0.25,
            0.3,
            5000,
        )
        self.previous_bbox = None
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

    def _detect_landmarks_in_crop(self, frame, detection):
        f_h, f_w, _ = frame.shape
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
            return None
        candidate = result.multi_face_landmarks[0]
        return np.array(
            [[x1 + p.x * crop_w, y1 + p.y * crop_h] for p in candidate.landmark],
            dtype=np.float32,
        )

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape
        # LatentSync's video readers already return RGB frames.
        frame = np.asarray(frame).astype(np.uint8)

        # YuNet is a dedicated full-frame face detector. Run FaceMesh only on
        # its padded crop, where small faces occupy enough of the input image.
        self.detector.setInputSize((f_w, f_h))
        _, detections = self.detector.detect(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if detections is not None:
            detections = list(detections)
            if self.previous_bbox is not None:
                px1, py1, px2, py2 = self.previous_bbox

                def intersection_over_union(item):
                    x1, y1, w, h = item[:4]
                    x2, y2 = x1 + w, y1 + h
                    intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(
                        0.0, min(y2, py2) - max(y1, py1)
                    )
                    union = w * h + (px2 - px1) * (py2 - py1) - intersection
                    return intersection / max(union, 1.0)

                best_iou = max(intersection_over_union(item) for item in detections)
                if best_iou > 0.05:
                    detections.sort(key=intersection_over_union, reverse=True)
                else:
                    detections.sort(key=lambda item: item[2] * item[3], reverse=True)
            else:
                detections.sort(key=lambda item: item[2] * item[3], reverse=True)
            for detection in detections:
                lmk = self._detect_landmarks_in_crop(frame, detection)
                if lmk is not None:
                    x1, y1 = np.floor(lmk.min(axis=0)).astype(int)
                    x2, y2 = np.ceil(lmk.max(axis=0)).astype(int)
                    w, h = x2 - x1, y2 - y1
                    if w > 0 and h > 0:
                        self.previous_bbox = (x1, y1, x2, y2)
                        x1, y1 = max(0, x1 - int(w * .05)), max(0, y1 - int(h * .05))
                        x2, y2 = min(f_w, x2 + int(w * .05)), min(f_h, y2 + int(h * .10))
                        return (x1, y1, x2, y2), np.round(lmk).astype(np.int32)

        # Keep the original full-frame path as a fallback for unusual cases
        # that YuNet does not propose.
        result = self.mesh.process(frame)
        if not result.multi_face_landmarks:
            # Retry at a larger resolution so that small faces occupy enough pixels
            # for FaceMesh's internal detector. Landmark coordinates are normalized,
            # so they can still be mapped directly to the original frame dimensions.
            enlarged_frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            result = self.mesh.process(enlarged_frame)
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
        lmk, (x1, y1, x2, y2) = best
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return None, None
        x1, y1 = max(0, x1 - int(w * .05)), max(0, y1 - int(h * .05))
        x2, y2 = min(f_w, x2 + int(w * .05)), min(f_h, y2 + int(h * .10))
        return (x1, y1, x2, y2), np.round(lmk).astype(np.int32)

    def close(self):
        self.mesh.close()
        self.crop_mesh.close()


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


