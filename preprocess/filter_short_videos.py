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

import os
import subprocess
from multiprocessing import Pool

import tqdm

from latentsync.utils.util import gather_video_paths_recursively

VIDEO_FPS = 25


def count_frames(video_path):
    """Frame count from the container metadata, falling back to duration * fps.

    We deliberately avoid `-count_frames`, which decodes the whole file just to count.
    """
    video_path_fixed = video_path.replace("\\", "/")
    command = (
        "ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames "
        f'-of default=noprint_wrappers=1:nokey=1 "{video_path_fixed}"'
    )
    nb_frames = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    if nb_frames.isdigit():
        return int(nb_frames)

    command = (
        "ffprobe -v error -show_entries format=duration "
        f'-of default=noprint_wrappers=1:nokey=1 "{video_path_fixed}"'
    )
    duration = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    return int(float(duration) * VIDEO_FPS)


def classify_segment(args):
    """Return (video_path, "short" | "unreadable" | "keep").

    Segmenting leaves a trailing remainder that is shorter than the requested segment time. Keeping
    it would skew training: UNetDataset picks a video uniformly at random and then a random window
    inside it, so a short segment gets its few frames sampled far more often than the rest of the
    footage. Anything below 3 * num_frames is silently skipped by the dataset anyway.

    A probe failure is reported separately rather than treated as "short": under a large worker
    pool ffprobe can fail transiently (file handles, an antivirus lock), and deleting a perfectly
    good segment because of that is not recoverable.
    """
    video_path, min_frames = args
    try:
        num_frames = count_frames(video_path)
    except Exception as e:
        print(f"Could not probe, leaving in place: {video_path} - {type(e).__name__}: {e}")
        return video_path, "unreadable"
    return video_path, "short" if num_frames < min_frames else "keep"


def filter_short_videos_multiprocessing(input_dir, num_workers, min_frames=125):
    video_paths = gather_video_paths_recursively(input_dir)
    tasks = [(video_path, min_frames) for video_path in video_paths]

    print(f"Filtering out segments shorter than {min_frames} frames in {input_dir} ...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm.tqdm(pool.imap_unordered(classify_segment, tasks), total=len(tasks)))
    else:
        results = [classify_segment(args) for args in tqdm.tqdm(tasks)]

    discarded = [video_path for video_path, verdict in results if verdict == "short"]
    unreadable = [video_path for video_path, verdict in results if verdict == "unreadable"]
    for video_path in discarded:
        os.remove(video_path)
        print(f"Discarded short segment: {video_path}")

    print(f"Discarded {len(discarded)} of {len(video_paths)} segments, {len(video_paths) - len(discarded)} remaining")
    if unreadable:
        print(f"{len(unreadable)} segments could not be probed and were left in place:")
        for video_path in unreadable:
            print(f"  {video_path}")


if __name__ == "__main__":
    input_dir = "my_data/segmented"
    num_workers = 8
    min_frames = 125

    filter_short_videos_multiprocessing(input_dir, num_workers, min_frames)
