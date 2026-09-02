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

import argparse
import os

VIDEO_FPS = 25
# Keep segments down to this fraction of a full one. The trailing remainder is what we want gone,
# and it is normally far shorter; the slack protects full segments that land a frame or two short
# because the source was variable frame rate and had to be conformed to 25 FPS.
SEGMENT_LENGTH_TOLERANCE = 0.9


def data_processing_pipeline(
    total_num_workers, per_gpu_num_workers, resolution, temp_dir, input_dir, segment_seconds=5
):
    print("Initializing components and loading libraries...")
    from preprocess.affine_transform import affine_transform_multi_gpus
    from preprocess.remove_broken_videos import remove_broken_videos_multiprocessing
    from preprocess.resample_fps_hz import resample_fps_hz_multiprocessing
    from preprocess.segment_videos import segment_videos_multiprocessing
    from preprocess.filter_short_videos import filter_short_videos_multiprocessing
    from preprocess.normalize_to_mp4 import normalize_to_mp4_multiprocessing

    print("Normalizing videos to mp4...")
    normalize_to_mp4_multiprocessing(input_dir, total_num_workers)

    print("Removing broken videos...")
    remove_broken_videos_multiprocessing(input_dir, total_num_workers)

    # Every segment must be the same length: UNetDataset samples a video uniformly at random and
    # then a random window inside it, so a short segment gets oversampled relative to the rest of
    # the footage. Forcing a keyframe every `segment_frames` lets the stream-copy segmenter cut
    # exactly on the requested boundaries, and the trailing remainder is dropped afterwards.
    segment_frames = segment_seconds * VIDEO_FPS

    print("Resampling FPS hz...")
    resampled_dir = os.path.join(os.path.dirname(input_dir), "resampled")
    resample_fps_hz_multiprocessing(input_dir, resampled_dir, total_num_workers, gop=segment_frames)

    # Shot detection is skipped: the source material is single-camera, so there are no cuts to split on.
    print("Segmenting videos...")
    segmented_dir = os.path.join(os.path.dirname(input_dir), "segmented")
    segment_videos_multiprocessing(resampled_dir, segmented_dir, total_num_workers, segment_time=segment_seconds)

    print("Filtering out short segments...")
    min_segment_frames = int(segment_frames * SEGMENT_LENGTH_TOLERANCE)
    filter_short_videos_multiprocessing(segmented_dir, total_num_workers, min_frames=min_segment_frames)

    # If there are too many videos, you can first use this step to filter and reduce the quantity
    # print("Filtering high resolution...")
    # high_resolution_dir = os.path.join(os.path.dirname(input_dir), "high_resolution")
    # filter_high_resolution_multiprocessing(segmented_dir, high_resolution_dir, resolution, total_num_workers)

    print("Affine transforming videos...")
    affine_transformed_dir = os.path.join(os.path.dirname(input_dir), "affine_transformed")
    affine_transform_multi_gpus(segmented_dir, affine_transformed_dir, temp_dir, resolution, max(1, per_gpu_num_workers // 2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_num_workers", type=int, default=100)
    parser.add_argument("--per_gpu_num_workers", type=int, default=20)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--segment_seconds", type=int, default=5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--input_dir", type=str, required=True)
    args = parser.parse_args()

    print("Preprocessing data...", flush=True)
    data_processing_pipeline(
        args.total_num_workers,
        args.per_gpu_num_workers,
        args.resolution,
        args.temp_dir,
        args.input_dir,
        args.segment_seconds,
    )
