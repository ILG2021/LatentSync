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
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, video_output])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def resample_fps_hz(video_input, video_output, gop):
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    video_input_fixed = video_input.replace("\\", "/")
    video_output_fixed = video_output.replace("\\", "/")

    # We always re-encode, even when the source is already 25 FPS / 16000 Hz. Stream copying would
    # keep the source GOP structure, and `segment_videos` can only cut on keyframes -- so without a
    # known keyframe interval the segments come out at arbitrary lengths.
    # -g fixes the NVENC GOP length, while -no-scenecut prevents adaptive I-frames from making the
    # stream-copy segments uneven. Encoding is offloaded to NVENC because resampling long 4K source
    # videos with libx264 otherwise leaves the GPU idle and can take hours.
    # -map 0:v:0 -map 0:a:0? : Only take first video and first audio stream
    # -sn -dn -map_metadata -1 : Strip subtitles, data streams, and all metadata
    print(f"Resampling/Re-encoding {video_input} (keyframe every {gop} frames)...")
    command = (
        f'ffmpeg -loglevel info -y -i "{video_input_fixed}" -map 0:v:0 -map 0:a:0? -r 25 '
        f"-c:v h264_nvenc -preset p5 -tune hq -rc vbr -cq 18 -b:v 0 "
        f"-g {gop} -no-scenecut 1 -forced-idr 1 "
        f'-c:a aac -ar 16000 -q:a 0 -sn -dn -map_metadata -1 -map_chapters -1 -ignore_unknown "{video_output_fixed}"'
    )

    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        # Report instead of raising: raised inside a Pool this would abort the whole stage and the
        # run could never get past the one bad file. Every neighbouring step logs and continues.
        print(f"FFmpeg failed with exit code {result.returncode} for file {video_input}")
        return video_input
    return None


def multi_run_wrapper(args):
    return resample_fps_hz(*args)


def resample_fps_hz_multiprocessing(input_dir, output_dir, num_workers, gop=125):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_paths(input_dir, output_dir)
    tasks = [(video_input, video_output, gop) for video_input, video_output in paths]

    print(f"Resampling FPS and Hz of {input_dir} ...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, tasks), total=len(tasks)))
    else:
        results = [resample_fps_hz(*args) for args in tqdm.tqdm(tasks)]

    failed = [video_input for video_input in results if video_input is not None]
    if failed:
        print(f"Failed to resample {len(failed)} of {len(tasks)} videos:")
        for video_input in failed:
            print(f"  {video_input}")


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/raw"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/resampled"
    num_workers = 20

    resample_fps_hz_multiprocessing(input_dir, output_dir, num_workers)
