# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from latentsync.utils.util import read_video, write_video
from torchvision import transforms
import cv2
from einops import rearrange
import torch
import numpy as np
from typing import Union
from .affine_transform import AlignRestore
from .face_detector import FaceDetector


# A face detector misses the occasional frame (a blink, a fast turn, a hand crossing the face).
# Dropping the whole clip for that would be wasteful, so short runs of missing landmarks are filled
# in from their neighbours. Longer runs mean the face really is gone and the clip is unusable.
MAX_LANDMARK_INTERPOLATION_GAP = 25

# MediaPipe's corresponding eyebrow/nose triangle is slightly larger than the
# InsightFace 106-point triangle used to prepare LatentSync's training crops.
# Contracting it around its centroid increases the aligned face scale while
# keeping the triangle centre fixed. This value was selected from paired-video
# measurements against the crop geometry used to train the checkpoint.
FACE_ANCHOR_SCALE = 0.935
LEFT_BROW_LANDMARKS = [105, 66]
RIGHT_BROW_LANDMARKS = [334, 296]
NOSE_LANDMARKS = [1, 4, 19, 94]


def alignment_anchors(face_landmarks: np.ndarray) -> np.ndarray:
    """Return LatentSync's three MediaPipe anchors, contracted about their centroid."""
    face_landmarks = np.asarray(face_landmarks, dtype=np.float32)
    if face_landmarks.ndim != 2 or face_landmarks.shape[1] != 2:
        raise ValueError(f"Expected landmarks shaped (N, 2), got {face_landmarks.shape}")

    anchors = np.asarray(
        [
            face_landmarks[LEFT_BROW_LANDMARKS].mean(axis=0),
            face_landmarks[RIGHT_BROW_LANDMARKS].mean(axis=0),
            face_landmarks[NOSE_LANDMARKS].mean(axis=0),
        ],
        dtype=np.float32,
    )
    centre = anchors.mean(axis=0, keepdims=True)
    anchors = centre + FACE_ANCHOR_SCALE * (anchors - centre)
    return np.round(anchors)


def interpolate_missing_landmarks(landmarks, max_gap: int = MAX_LANDMARK_INTERPOLATION_GAP):
    """Fill short runs of `None` in `landmarks` in place, by interpolating between the neighbours.

    Returns (failed_index, repaired_count). `failed_index` is the first frame of the first gap that
    was too long to fill, or None when every gap could be repaired.
    """
    valid_indices = [index for index, value in enumerate(landmarks) if value is not None]
    if not valid_indices:
        return 0, 0

    repaired_count = len(landmarks) - len(valid_indices)
    missing_start = None
    for index in range(len(landmarks) + 1):
        is_missing = index < len(landmarks) and landmarks[index] is None
        if is_missing and missing_start is None:
            missing_start = index
        elif not is_missing and missing_start is not None:
            if index - missing_start > max_gap:
                return missing_start, repaired_count

            left_index = missing_start - 1
            right_index = index if index < len(landmarks) else None
            for missing_index in range(missing_start, index):
                if left_index < 0:
                    landmarks[missing_index] = landmarks[right_index].copy()
                elif right_index is None:
                    landmarks[missing_index] = landmarks[left_index].copy()
                else:
                    weight = (missing_index - left_index) / (right_index - left_index)
                    landmarks[missing_index] = (
                        landmarks[left_index] * (1.0 - weight) + landmarks[right_index] * weight
                    )
            missing_start = None

    return None, repaired_count


def load_fixed_mask(resolution: int, mask_image_path="latentsync/utils/mask.png") -> torch.Tensor:
    mask_image = cv2.imread(mask_image_path)
    mask_image = cv2.cvtColor(mask_image, cv2.COLOR_BGR2RGB)
    mask_image = cv2.resize(mask_image, (resolution, resolution), interpolation=cv2.INTER_LANCZOS4) / 255.0
    mask_image = rearrange(torch.from_numpy(mask_image), "h w c -> c h w")
    return mask_image


class ImageProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu", mask_image=None):
        self.resolution = resolution
        self.resize = transforms.Resize(
            (resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True
        )
        self.normalize = transforms.Normalize([0.5], [0.5], inplace=True)
        self.restorer = AlignRestore(resolution=resolution, device=device)

        if mask_image is None:
            self.mask_image = load_fixed_mask(resolution)
        else:
            self.mask_image = mask_image

        if device == "cpu":
            self.face_detector = None
        else:
            self.face_detector = FaceDetector(device=device)

    def affine_transform(self, image: torch.Tensor) -> np.ndarray:
        face_landmarks = self.detect_face_landmarks(image)
        if face_landmarks is None:
            raise RuntimeError("Face not detected")
        return self.affine_transform_with_landmarks(image, face_landmarks)

    def detect_face_landmarks(self, image: torch.Tensor):
        if self.face_detector is None:
            raise NotImplementedError("Using the CPU for face detection is not supported")
        _, face_landmarks = self.face_detector(image)
        return face_landmarks

    def reset_face_tracking(self):
        if self.face_detector is not None:
            self.face_detector.reset_tracking()

    def release_face_detector(self):
        """Release face-detection models once affine preprocessing is complete."""
        if self.face_detector is not None:
            self.face_detector.close()
            self.face_detector = None

    def affine_transform_with_landmarks(self, image: torch.Tensor, face_landmarks: np.ndarray) -> np.ndarray:
        # The UNet checkpoint was trained on crops built from InsightFace 106-point landmarks:
        #   left  = mean(lmk106[[43, 48, 49, 51, 50]])   # left eyebrow center
        #   right = mean(lmk106[101:106])                # right eyebrow center
        #   nose  = mean(lmk106[[74, 77, 83, 86]])       # nose center
        # InsightFace is not licensed for commercial use, so the detector here is YuNet +
        # MediaPipe FaceMesh. These index sets are the MediaPipe points whose means land on the
        # same anatomical positions. Their triangle is then contracted around its centroid because
        # paired-video measurements show that the raw MediaPipe triangle produces a consistently
        # smaller aligned face than the checkpoint's InsightFace-based training crops.
        #
        # These are eyebrow centers, not eye centers -- the names are kept from upstream. Using
        # eye centers instead shrinks the eye/brow-to-nose span and zooms the crop in by ~31%,
        # which is far outside what the checkpoint was trained on.
        landmarks3 = alignment_anchors(face_landmarks)

        face, affine_matrix = self.restorer.align_warp_face(image.copy(), landmarks3=landmarks3, smooth=True)
        box = [0, 0, face.shape[1], face.shape[0]]  # x1, y1, x2, y2
        face = cv2.resize(face, (self.resolution, self.resolution), interpolation=cv2.INTER_LANCZOS4)
        face = rearrange(torch.from_numpy(face), "h w c -> c h w")
        return face, box, affine_matrix

    def preprocess_fixed_mask_image(self, image: torch.Tensor, affine_transform=False):
        if affine_transform:
            image, _, _ = self.affine_transform(image)
        else:
            image = self.resize(image)
        pixel_values = self.normalize(image / 255.0)
        masked_pixel_values = pixel_values * self.mask_image
        return pixel_values, masked_pixel_values, self.mask_image[0:1]

    def prepare_masks_and_masked_images(self, images: Union[torch.Tensor, np.ndarray], affine_transform=False):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")

        results = [self.preprocess_fixed_mask_image(image, affine_transform=affine_transform) for image in images]

        pixel_values_list, masked_pixel_values_list, masks_list = list(zip(*results))
        return torch.stack(pixel_values_list), torch.stack(masked_pixel_values_list), torch.stack(masks_list)

    def process_images(self, images: Union[torch.Tensor, np.ndarray]):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)
        if images.shape[3] == 3:
            images = rearrange(images, "f h w c -> f c h w")
        images = self.resize(images)
        pixel_values = self.normalize(images / 255.0)
        return pixel_values


class VideoProcessor:
    def __init__(self, resolution: int = 512, device: str = "cpu"):
        self.image_processor = ImageProcessor(resolution, device)

    def affine_transform_video(self, video_path):
        video_frames = read_video(video_path, change_fps=False)

        # Detect the whole track first, then repair it, then align: the same order the inference
        # pipeline uses. Aligning frame by frame would abort the clip on the first missed face.
        self.image_processor.reset_face_tracking()
        landmarks = [self.image_processor.detect_face_landmarks(frame) for frame in video_frames]
        failed_index, repaired_count = interpolate_missing_landmarks(landmarks)
        if failed_index is not None:
            raise RuntimeError(
                f"Face not detected for more than {MAX_LANDMARK_INTERPOLATION_GAP} consecutive "
                f"frames, starting at frame {failed_index}"
            )
        if repaired_count:
            print(f"Interpolated face landmarks for {repaired_count} missing frames: {video_path}")

        # AlignRestore keeps a running translation offset to damp jitter. It has to start fresh for
        # every clip, otherwise the first frames get pulled towards the previous clip's face.
        self.image_processor.restorer.p_bias = None

        results = []
        for frame, face_landmarks in zip(video_frames, landmarks):
            face, _, _ = self.image_processor.affine_transform_with_landmarks(frame, face_landmarks)
            results.append(face)
        results = torch.stack(results)

        results = rearrange(results, "f c h w -> f h w c").numpy()
        return results


if __name__ == "__main__":
    video_processor = VideoProcessor(256, "cuda")
    video_frames = video_processor.affine_transform_video("assets/demo2_video.mp4")
    write_video("output.mp4", video_frames, fps=25)
