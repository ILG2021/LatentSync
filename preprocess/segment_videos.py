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
import math
import re
import subprocess
import tqdm
from multiprocessing import Pool

paths = []
SEGMENT_LENGTH_TOLERANCE = 0.9


def retained_segment_count(video_input, segment_time):
    """Return the number of segments expected to remain after short-tail filtering."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_input,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    duration = float(result.stdout.strip())
    # Allow a small timestamp rounding error around exact segment boundaries.
    full_segments = int(math.floor((duration + 0.05) / segment_time))
    remainder = max(0., duration - full_segments * segment_time)
    keep_tail = remainder >= segment_time * SEGMENT_LENGTH_TOLERANCE
    return full_segments + int(keep_tail)


def segments_already_complete(video_input, output_dir, video_basename, segment_time):
    if not os.path.isdir(output_dir):
        return False
    try:
        expected_count = retained_segment_count(video_input, segment_time)
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    if expected_count == 0:
        return False

    pattern = re.compile(rf"^{re.escape(video_basename)}_(\d{{3,}})\.mp4$", re.IGNORECASE)
    existing_indices = {
        int(match.group(1))
        for filename in os.listdir(output_dir)
        if (match := pattern.match(filename))
        and os.path.getsize(os.path.join(output_dir, filename)) > 0
    }
    return all(index in existing_indices for index in range(expected_count))


def gather_paths(input_dir, output_dir, segment_time):
    for video in sorted(os.listdir(input_dir)):
        if video.lower().endswith(".mp4"):
            video_basename = video[:-4]
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, f"{video_basename}_%03d.mp4")
            if segments_already_complete(video_input, output_dir, video_basename, segment_time):
                print(f"Skipping already segmented video: {video_input}")
                continue
            paths.append([video_input, video_output])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(
                os.path.join(input_dir, video),
                os.path.join(output_dir, video),
                segment_time,
            )


def segment_video(video_input, video_output, segment_time):
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    video_input_fixed = video_input.replace("\\", "/")
    video_output_fixed = video_output.replace("\\", "/")
    # -c:v copy can only cut on keyframes. `resample_fps_hz` forces one every `segment_time` seconds,
    # so the cuts land exactly on the requested boundaries. The trailing remainder is shorter than
    # segment_time and gets dropped by `filter_short_videos`.
    command = (
        f'ffmpeg -nostdin -loglevel info -y -i "{video_input_fixed}" -map 0 -c:v copy '
        f'-segment_time {segment_time} -f segment -reset_timestamps 1 -q:a 0 "{video_output_fixed}"'
    )
    subprocess.run(command, shell=True)


def multi_run_wrapper(args):
    return segment_video(*args)


def segment_videos_multiprocessing(input_dir, output_dir, num_workers, segment_time=5):
    print(f"Recursively gathering video paths of {input_dir} ...")
    paths.clear()
    gather_paths(input_dir, output_dir, segment_time)
    tasks = [(video_input, video_output, segment_time) for video_input, video_output in paths]

    print(f"Segmenting videos of {input_dir} ...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, tasks), total=len(tasks)):
                pass
    else:
        for args in tqdm.tqdm(tasks):
            segment_video(*args)


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/shot"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/segmented"
    num_workers = 50

    segment_videos_multiprocessing(input_dir, output_dir, num_workers)
