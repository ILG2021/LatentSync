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
import tqdm
from multiprocessing import Pool

paths = []


def gather_paths(input_dir, output_dir):
    for video in sorted(os.listdir(input_dir)):
        if video.lower().endswith(".mp4"):
            video_basename = video[:-4]
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, f"{video_basename}_%03d.mp4")
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, video_output])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def segment_video(video_input, video_output, segment_time):
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    video_input_fixed = video_input.replace("\\", "/")
    video_output_fixed = video_output.replace("\\", "/")
    # -c:v copy can only cut on keyframes. `resample_fps_hz` forces one every `segment_time` seconds,
    # so the cuts land exactly on the requested boundaries. The trailing remainder is shorter than
    # segment_time and gets dropped by `filter_short_videos`.
    command = (
        f'ffmpeg -loglevel info -y -i "{video_input_fixed}" -map 0 -c:v copy '
        f'-segment_time {segment_time} -f segment -reset_timestamps 1 -q:a 0 "{video_output_fixed}"'
    )
    subprocess.run(command, shell=True)


def multi_run_wrapper(args):
    return segment_video(*args)


def segment_videos_multiprocessing(input_dir, output_dir, num_workers, segment_time=5):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_paths(input_dir, output_dir)
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
