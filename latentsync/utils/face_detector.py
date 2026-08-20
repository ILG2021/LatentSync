import mediapipe as mp
import numpy as np
import torch

class FaceDetector:
    def __init__(self, device="cuda"):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def __call__(self, frame, threshold=0.5):
        f_h, f_w, _ = frame.shape
        # LatentSync's video readers already return RGB frames.
        result = self.mesh.process(np.asarray(frame).astype(np.uint8))
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
        if w < 50 or h < 80 or w / max(h, 1) > 1.5 or w / max(h, 1) < 0.2:
            return None, None
        x1, y1 = max(0, x1 - int(w * .05)), max(0, y1 - int(h * .05))
        x2, y2 = min(f_w, x2 + int(w * .05)), min(f_h, y2 + int(h * .10))
        return (x1, y1, x2, y2), np.round(lmk).astype(np.int32)

    def close(self):
        self.mesh.close()


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


