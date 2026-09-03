"""GPU-capable implementations of the face models used during preprocessing.

The PyPI builds of OpenCV and MediaPipe execute YuNet and FaceMesh on the CPU.
These small wrappers keep the public behaviour of those libraries while allowing
the models to use ONNX Runtime's CUDA execution provider.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_ort_session(model_path: Path, device_id: int):
    """Create a CUDA ORT session, raising instead of silently using the CPU."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
        raise RuntimeError(
            "onnxruntime has no CUDAExecutionProvider (available providers: "
            f"{available}). Install onnxruntime-gpu, not onnxruntime."
        )

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=[
            ("CUDAExecutionProvider", {"device_id": device_id}),
            "CPUExecutionProvider",
        ],
    )
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise RuntimeError(f"ONNX Runtime did not select CUDA: {session.get_providers()}")
    return session


class _ScaledYuNet:
    """Shared aspect-preserving resize and coordinate mapping for YuNet."""

    _STRIDES = (8, 16, 32)

    def __init__(self, max_input_size: int):
        if max_input_size < 32:
            raise ValueError("YuNet max_input_size must be at least 32")
        self.max_input_size = max_input_size

    @staticmethod
    def _resolve_score_threshold(value, default):
        threshold = default if value is None else float(value)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("YuNet score threshold must be between 0 and 1")
        return threshold

    @staticmethod
    def _prepare_input(rgb_image, maximum: int):
        height, width = rgb_image.shape[:2]
        scale = min(1.0, maximum / max(width, height)) if maximum > 0 else 1.0
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        if resized_w != width or resized_h != height:
            resized = cv2.resize(rgb_image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        else:
            resized = rgb_image

        # Match FaceDetectorYN: preserve the image aspect ratio, then pad only
        # the right and bottom edges for YuNet's stride-32 heads.
        net_w = max(32, int(np.ceil(resized_w / 32.0)) * 32)
        net_h = max(32, int(np.ceil(resized_h / 32.0)) * 32)
        if net_w == resized_w and net_h == resized_h:
            return resized, scale
        padded = np.zeros((net_h, net_w, 3), dtype=resized.dtype)
        padded[:resized_h, :resized_w] = resized
        return padded, scale

    @staticmethod
    def _restore_coordinates(detections, scale):
        detections[:, :14] /= scale
        return detections


class OpenCvYuNet(_ScaledYuNet):
    """OpenCV CPU fallback that also avoids running YuNet on the full 4K frame."""

    def __init__(
        self,
        model_path: Path,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_input_size: int = 640,
    ):
        super().__init__(max_input_size)
        self.score_threshold = self._resolve_score_threshold(score_threshold, 0.25)
        self._active_score_threshold = self.score_threshold
        self.detector = cv2.FaceDetectorYN.create(
            str(model_path), "", (320, 320), self.score_threshold, nms_threshold, top_k
        )

    def detect(self, rgb_image, score_threshold=None):
        effective_threshold = self._resolve_score_threshold(
            score_threshold, self.score_threshold
        )
        if effective_threshold != self._active_score_threshold:
            self.detector.setScoreThreshold(effective_threshold)
            self._active_score_threshold = effective_threshold
        image, scale = self._prepare_input(rgb_image, self.max_input_size)
        net_h, net_w = image.shape[:2]
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.detector.setInputSize((net_w, net_h))
        retval, detections = self.detector.detect(bgr)
        if detections is not None:
            detections = self._restore_coordinates(detections, scale)
        return retval, detections


class OrtYuNet(_ScaledYuNet):
    """YuNet inference and post-processing with an OpenCV FaceDetectorYN-like API."""

    def __init__(
        self,
        model_path: Path,
        device_id: int,
        score_threshold: float = 0.25,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_input_size: int = 640,
    ):
        super().__init__(max_input_size)
        self.session = create_ort_session(model_path, device_id)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = {output.name for output in self.session.get_outputs()}
        required = {
            f"{kind}_{stride}"
            for kind in ("cls", "obj", "bbox", "kps")
            for stride in self._STRIDES
        }
        missing = required - self.output_names
        if missing:
            raise RuntimeError(f"Unexpected YuNet model outputs; missing {sorted(missing)}")
        self.score_threshold = self._resolve_score_threshold(score_threshold, 0.25)
        self.nms_threshold = nms_threshold
        self.top_k = top_k

    def detect(self, rgb_image, score_threshold=None):
        effective_threshold = self._resolve_score_threshold(
            score_threshold, self.score_threshold
        )
        image, scale = self._prepare_input(rgb_image, self.max_input_size)
        net_h, net_w = image.shape[:2]
        # Convert only the reduced image rather than the complete 4K frame.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # YuNet expects unnormalised BGR values. This matches FaceDetectorYN's
        # blobFromImage(padded_image) preprocessing.
        blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(net_w, net_h), swapRB=False)
        requested = [
            f"{kind}_{stride}"
            for kind in ("cls", "obj", "bbox", "kps")
            for stride in self._STRIDES
        ]
        values = self.session.run(requested, {self.input_name: blob})
        outputs = dict(zip(requested, values))

        rows = []
        for stride in self._STRIDES:
            cols = net_w // stride
            cls = outputs[f"cls_{stride}"].reshape(-1)
            obj = outputs[f"obj_{stride}"].reshape(-1)
            bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)
            kps = outputs[f"kps_{stride}"].reshape(-1, 10)
            scores = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))
            for index in np.flatnonzero(scores >= effective_threshold):
                row, col = divmod(int(index), cols)
                center_x = (col + bbox[index, 0]) * stride
                center_y = (row + bbox[index, 1]) * stride
                width = np.exp(np.clip(bbox[index, 2], -20, 20)) * stride
                height = np.exp(np.clip(bbox[index, 3], -20, 20)) * stride
                landmarks = []
                for point in range(5):
                    landmarks.extend(
                        ((col + kps[index, point * 2]) * stride,
                         (row + kps[index, point * 2 + 1]) * stride)
                    )
                rows.append(
                    [center_x - width / 2, center_y - height / 2, width, height]
                    + landmarks
                    + [float(scores[index])]
                )

        if not rows:
            return 1, None
        detections = np.asarray(rows, dtype=np.float32)
        keep = cv2.dnn.NMSBoxes(
            detections[:, :4].tolist(),
            detections[:, 14].tolist(),
            effective_threshold,
            self.nms_threshold,
            top_k=self.top_k,
        )
        if len(keep) == 0:
            return 1, None
        detections = detections[np.asarray(keep).reshape(-1)]
        return 1, self._restore_coordinates(detections, scale)


class OrtFaceLandmark:
    """The 478-point MediaPipe Face Landmarker model running through ORT CUDA."""

    def __init__(self, model_path: Path, device_id: int):
        self.session = create_ort_session(model_path, device_id)
        self.input = self.session.get_inputs()[0]
        outputs = self.session.get_outputs()
        input_dims = [dimension for dimension in self.input.shape if isinstance(dimension, int)]
        if len(input_dims) < 3:
            raise RuntimeError(f"Unexpected landmark input shape: {self.input.shape}")
        self.input_size = max(input_dims[-3:])
        self.nchw = len(self.input.shape) == 4 and self.input.shape[1] == 3

        landmark_outputs = [
            output
            for output in outputs
            if 1434 in [dimension for dimension in output.shape if isinstance(dimension, int)]
        ]
        if len(landmark_outputs) != 1:
            raise RuntimeError(
                "Expected one 478x3 landmark output, got "
                f"{[(output.name, output.shape) for output in outputs]}"
            )
        self.landmark_output = landmark_outputs[0].name
        # The pinned MediaPipe v2 conversion retains this output name. Selecting it
        # explicitly avoids confusing it with the model's second scalar output.
        presence_outputs = [output for output in outputs if output.name == "Identity_1"]
        if len(presence_outputs) != 1:
            raise RuntimeError(
                "Expected MediaPipe face-presence output 'Identity_1', got "
                f"{[output.name for output in outputs]}"
            )
        self.presence_output = presence_outputs[0].name

    def process(self, rgb_frame, detection):
        """Run the landmark model on a MediaPipe-style oriented face ROI.

        YuNet's first two keypoints are the eyes.  Rotating the ROI to make that
        line horizontal is important: the standalone landmark model normally
        receives this transform from the MediaPipe graph.
        """
        x, y, width, height = detection[:4]
        eye_0 = np.asarray(detection[4:6], dtype=np.float32)
        eye_1 = np.asarray(detection[6:8], dtype=np.float32)
        angle = np.degrees(np.arctan2(eye_1[1] - eye_0[1], eye_1[0] - eye_0[0]))
        roi_size = max(max(float(width), float(height)) * 1.5, 1.0)
        center = np.array([x + width / 2, y + height / 2], dtype=np.float32)

        matrix = cv2.getRotationMatrix2D(tuple(center), float(angle), self.input_size / roi_size)
        mapped_center = matrix[:, :2] @ center + matrix[:, 2]
        matrix[:, 2] += self.input_size / 2 - mapped_center
        resized = cv2.warpAffine(
            rgb_frame,
            matrix,
            (self.input_size, self.input_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        # Face Landmarker v2 uses approximately [-1, 1] RGB input.
        tensor = (resized.astype(np.float32) - 127.0) / 128.0
        if self.nchw:
            tensor = np.transpose(tensor, (2, 0, 1))
        tensor = tensor[None]
        raw, presence_logit = self.session.run(
            [self.landmark_output, self.presence_output], {self.input.name: tensor}
        )
        presence_value = float(np.asarray(presence_logit).reshape(-1)[0])
        presence = (
            presence_value
            if 0.0 <= presence_value <= 1.0
            else 1.0 / (1.0 + np.exp(-np.clip(presence_value, -30.0, 30.0)))
        )
        if presence < 0.5:
            return None
        landmarks = raw.reshape(-1, 3)
        if len(landmarks) != 478:
            raise RuntimeError(f"Face landmark model returned {len(landmarks)} points, expected 478")
        landmarks = landmarks[:, :2].astype(np.float32)
        # Some exports expose normalized coordinates while the official graph exposes
        # coordinates in input pixels. Support both without changing the public output.
        if np.nanpercentile(np.abs(landmarks), 99) <= 2.0:
            landmarks *= self.input_size
        inverse = cv2.invertAffineTransform(matrix)
        return cv2.transform(landmarks[None], inverse)[0]
