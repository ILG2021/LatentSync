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
import shutil

paths = []


def gather_paths(input_dir, output_dir):
    for video in sorted(os.listdir(input_dir)):
        if video.endswith(".mp4"):
            video_input = os.path.join(input_dir, video)
            video_output = os.path.join(output_dir, video)
            if os.path.isfile(video_output):
                continue
            paths.append([video_input, video_output])
        elif os.path.isdir(os.path.join(input_dir, video)):
            gather_paths(os.path.join(input_dir, video), os.path.join(output_dir, video))


def get_video_audio_info(video_path):
    video_input_fixed = video_path.replace("\\", "/")
    # Get FPS
    fps_command = f'ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "{video_input_fixed}"'
    try:
        fps_raw = subprocess.check_output(fps_command, shell=True).decode('utf-8').strip()
        if '/' in fps_raw:
            num, den = fps_raw.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 0
        else:
            fps = float(fps_raw) if fps_raw else 0
    except Exception:
        fps = 0

    # Get Sample Rate
    sr_command = f'ffprobe -v error -select_streams a:0 -show_entries stream=sample_rate -of default=noprint_wrappers=1:nokey=1 "{video_input_fixed}"'
    try:
        sample_rate = int(subprocess.check_output(sr_command, shell=True).decode('utf-8').strip())
    except Exception:
        sample_rate = 0
    
    return fps, sample_rate


def resample_fps_hz(video_input, video_output):
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    video_input_fixed = video_input.replace("\\", "/")
    video_output_fixed = video_output.replace("\\", "/")
    
    fps, sr = get_video_audio_info(video_input)
    
    # Even if parameters match, we use ffmpeg to 'clean' the streams (strip metadata/broken streams)
    if round(fps) == 25 and sr == 16000:
        print(f"Cleaning streams for {video_input} (already 25fps, 16kHz)")
        # Use -c copy with mapping to strip problematic extra streams while keeping it fast
        command = f'ffmpeg -loglevel info -y -i "{video_input_fixed}" -map 0:v -map 0:a? -c copy "{video_output_fixed}"'
        subprocess.run(command, shell=True)
        return

    # Use loglevel info to show progress. 
    # Use -map 0:v -map 0:a? to only include video and audio, ignoring problematic metadata/data streams.
    command = f'ffmpeg -loglevel info -y -i "{video_input_fixed}" -map 0:v -map 0:a? -r 25 -c:v libx264 -preset fast -crf 18 -c:a aac -ar 16000 -q:a 0 "{video_output_fixed}"'
    subprocess.run(command, shell=True)


def multi_run_wrapper(args):
    return resample_fps_hz(*args)


def resample_fps_hz_multiprocessing(input_dir, output_dir, num_workers):
    print(f"Recursively gathering video paths of {input_dir} ...")
    gather_paths(input_dir, output_dir)

    print(f"Resampling FPS and Hz of {input_dir} ...")
    if num_workers > 1:
        with Pool(num_workers) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
                pass
    else:
        for args in tqdm.tqdm(paths):
            resample_fps_hz(*args)


if __name__ == "__main__":
    input_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/raw"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/VoxCeleb2/resampled"
    num_workers = 20

    resample_fps_hz_multiprocessing(input_dir, output_dir, num_workers)
