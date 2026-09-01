# Copyright (c) 2026
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
#
# Face alignment and paste-back utilities.
#
# The point-set alignment below is the classical Procrustes / Kabsch-Umeyama
# similarity fit, described in:
#   S. Umeyama, "Least-Squares Estimation of Transformation Parameters Between
#   Two Point Patterns", IEEE TPAMI 13(4), 1991.
# The paste-back path is ordinary alpha compositing: inverse-warp the generated
# crop, erode the warped support mask to drop resampling fringe, feather the
# result with a Gaussian, then blend over the source frame.

import cv2
import kornia
import numpy as np
import torch
from einops import rearrange


# Reference landmark layout, expressed in the 75 x 100 canonical face box that
# the crop geometry is defined in. Rows are (left eye, right eye, nose).
#
# WARNING: these constants, the 75 x 100 box and CROP_SCALE together define the
# exact crop the UNet checkpoint was trained on. Changing any of them silently
# degrades generation quality -- treat them as weights, not as tunables.
_REFERENCE_LANDMARKS = np.array(
    [[17.0, 20.0], [58.0, 20.0], [37.5, 40.0]],
    dtype=np.float64,
)
_REFERENCE_BOX = (75.0, 100.0)  # (width, height)
_CROP_SCALE = 2.8 / 256.0  # multiplied by the target resolution

# Exponential moving average factor for the inter-frame jitter damping applied
# to the crop translation. 0.0 disables damping, 1.0 freezes the first offset.
_SMOOTHING_MOMENTUM = 0.2


def _solve_similarity_transform(source_points, target_points):
    """Fit the 2x3 similarity transform mapping ``source_points`` onto ``target_points``.

    Returns the transform together with the mean-centred, scale-normalised
    copies of both point sets, which the caller reuses for jitter damping.
    """
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    if source.shape != target.shape:
        raise ValueError(f"Point set shapes differ: {source.shape} vs {target.shape}")

    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centred = source - source_centroid
    target_centred = target - target_centroid

    # Isotropic spread of each point cloud; their ratio is the similarity scale.
    source_spread = np.std(source_centred, ddof=1)
    target_spread = np.std(target_centred, ddof=1)
    if source_spread < np.finfo(np.float64).eps:
        raise ValueError("Degenerate landmarks: source points are coincident")

    source_normalized = source_centred / source_spread
    target_normalized = target_centred / target_spread

    # Optimal rotation from the SVD of the cross-covariance, with the standard
    # reflection guard so the fit stays a proper rotation.
    u, _, vt = np.linalg.svd(source_normalized.T @ target_normalized)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T

    scale = target_spread / source_spread
    translation = target_centroid - scale * (rotation @ source_centroid)

    transform = np.empty((2, 3), dtype=np.float64)
    transform[:, :2] = scale * rotation
    transform[:, 2] = translation
    return transform, source_normalized, target_normalized


class AlignRestore:
    """Crops a face to the canonical frame and pastes a generated crop back."""

    def __init__(self, align_points=3, resolution=256, device="cpu", dtype=torch.float16):
        if align_points != 3:
            raise ValueError(f"Only 3-point alignment is supported, got align_points={align_points}")

        self.device = device
        self.dtype = dtype
        self.upscale_factor = 1

        crop_ratio = resolution * _CROP_SCALE
        self.crop_ratio = (crop_ratio, crop_ratio)
        self.face_template = _REFERENCE_LANDMARKS * crop_ratio
        self.face_size = (
            int(_REFERENCE_BOX[0] * crop_ratio),
            int(_REFERENCE_BOX[1] * crop_ratio),
        )

        # Running translation offset used to damp inter-frame jitter. Reset it
        # to None before starting a new shot.
        self.p_bias = None

        self.fill_value = torch.tensor([127, 127, 127], device=device, dtype=dtype)
        self.mask = torch.ones((1, 1, self.face_size[1], self.face_size[0]), device=device, dtype=dtype)

    def transformation_from_points(self, points1, points0, smooth=True, p_bias=None):
        """Fit the crop transform, optionally damping its translation over time."""
        transform, source_normalized, target_normalized = _solve_similarity_transform(points1, points0)

        if smooth:
            offset = target_normalized[2] - source_normalized[2]
            if p_bias is not None:
                offset = p_bias * _SMOOTHING_MOMENTUM + offset * (1.0 - _SMOOTHING_MOMENTUM)
            p_bias = offset
            transform[:, 2] += offset

        return transform.astype(np.float32), p_bias

    def align_warp_face(self, img, landmarks3, smooth=True):
        """Crop the canonical face box out of ``img`` given three landmarks."""
        affine_matrix, self.p_bias = self.transformation_from_points(
            landmarks3, self.face_template, smooth, self.p_bias
        )

        img = rearrange(torch.from_numpy(img).to(device=self.device, dtype=self.dtype), "h w c -> c h w").unsqueeze(0)
        affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)

        cropped_face = kornia.geometry.transform.warp_affine(
            img,
            affine_matrix,
            (self.face_size[1], self.face_size[0]),
            mode="bilinear",
            padding_mode="fill",
            fill_value=self.fill_value,
        )
        cropped_face = rearrange(cropped_face.squeeze(0), "c h w -> h w c").cpu().numpy().astype(np.uint8)
        return cropped_face, affine_matrix

    def restore_img(self, input_img, face, affine_matrix):
        """Composite a generated face crop back onto the full source frame."""
        height, width, _ = input_img.shape

        if isinstance(affine_matrix, np.ndarray):
            affine_matrix = torch.from_numpy(affine_matrix).to(device=self.device, dtype=self.dtype).unsqueeze(0)
        inverse_matrix = kornia.geometry.transform.invert_affine_transform(affine_matrix)

        # Warp the crop back into frame coordinates and undo the [-1, 1] scaling.
        face = face.to(dtype=self.dtype).unsqueeze(0)
        warped_face = kornia.geometry.transform.warp_affine(
            face, inverse_matrix, (height, width), mode="bilinear", padding_mode="fill", fill_value=self.fill_value
        ).squeeze(0)
        warped_face = (warped_face / 2 + 0.5).clamp(0, 1) * 255

        # Support of the warped crop, shrunk slightly so bilinear resampling
        # fringe along the crop border does not leak into the composite.
        warped_mask = kornia.geometry.transform.warp_affine(
            self.mask, inverse_matrix, (height, width), padding_mode="zeros"
        )
        border_kernel = torch.ones(
            (int(2 * self.upscale_factor), int(2 * self.upscale_factor)), device=self.device, dtype=self.dtype
        )
        warped_mask = kornia.morphology.erosion(warped_mask, border_kernel)

        pasted_face = warped_mask.squeeze(0).expand_as(warped_face) * warped_face

        # Feather width scales with the face size so the seam stays proportional.
        face_area = torch.sum(warped_mask.float())
        edge_width = max(int(face_area.item() ** 0.5) // 20, 1)
        erosion_radius = edge_width * 2

        # Done on CPU: an equivalent kornia erosion at this radius allocates an
        # intermediate of (radius^2, 1, H, W) and exhausts GPU memory.
        mask_np = warped_mask.squeeze().cpu().numpy().astype(np.float32)
        mask_np = cv2.erode(mask_np, np.ones((erosion_radius, erosion_radius), np.uint8))
        inner_mask = torch.from_numpy(mask_np).to(device=self.device, dtype=self.dtype)[None, None, ...]

        # OpenCV's sigma-from-kernel-size rule, see cv2.getGaussianKernel docs.
        blur_size = edge_width * 2 + 1
        sigma = 0.3 * ((blur_size - 1) * 0.5 - 1) + 0.8
        alpha = kornia.filters.gaussian_blur2d(inner_mask, (blur_size, blur_size), (sigma, sigma)).squeeze(0)
        alpha = alpha.expand_as(warped_face)

        input_img = rearrange(torch.from_numpy(input_img).to(device=self.device, dtype=self.dtype), "h w c -> c h w")
        blended = alpha * pasted_face + (1 - alpha) * input_img

        blended = rearrange(blended, "c h w -> h w c").contiguous().to(dtype=torch.uint8)
        return blended.cpu().numpy()
