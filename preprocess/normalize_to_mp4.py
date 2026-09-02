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
import shutil
import subprocess
from multiprocessing import Pool

import tqdm

from latentsync.utils.util import check_ffmpeg_installed

# Everything downstream of this step looks for ".mp4" only, so any other container here would be
# skipped without a word. Convert them instead of letting footage disappear silently.
CONVERTIBLE_EXTENSIONS = (
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".flv",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".ts",
    ".mts",
    ".m2ts",
)


def gather_convertible_paths_recursively(input_dir):
    print(f"Recursively gathering non-mp4 video paths of {input_dir} ...")
    paths = []
    gather_convertible_paths(input_dir, paths)
    return paths


def gather_convertible_paths(input_dir, paths):
    for file in sorted(os.listdir(input_dir)):
        filepath = os.path.join(input_dir, file)
        if file.lower().endswith(CONVERTIBLE_EXTENSIONS):
            paths.append(filepath)
        elif os.path.isdir(filepath):
            gather_convertible_paths(filepath, paths)


def convert_to_mp4(args):
    source_path, input_dir, originals_dir = args
    mp4_path = source_path.rsplit(".", 1)[0] + ".mp4"
    if os.path.isfile(mp4_path):
        print(f"Skipping {source_path}: {mp4_path} already exists")
        return

    # -crf 18 -preset slow: visually lossless. This is a lossy re-encode, so the original is kept.
    command = (
        f'ffmpeg -y -i "{source_path}" -c:v libx264 -crf 18 -preset slow '
        f'-c:a aac -b:a 192k -strict experimental -loglevel info "{mp4_path}"'
    )
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {source_path}: {e}")
        return

    # Move the source out of the dataset rather than deleting it: the mp4 is a lossy re-encode and
    # `input_dir` may be the only copy of the original footage.
    # Mirror the layout under input_dir instead of flattening to a basename: two sources with the
    # same name in different subdirectories would otherwise overwrite each other here.
    destination = os.path.join(originals_dir, os.path.relpath(source_path, input_dir))
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.move(source_path, destination)


def normalize_to_mp4_multiprocessing(input_dir, num_workers, originals_dir=None):
    check_ffmpeg_installed()
    source_paths = gather_convertible_paths_recursively(input_dir)
    if not source_paths:
        print("No non-mp4 videos found.")
        return

    if originals_dir is None:
        originals_dir = os.path.join(os.path.dirname(os.path.abspath(input_dir)), "converted_originals")
    tasks = [(source_path, input_dir, originals_dir) for source_path in source_paths]

    print(f"Converting {len(tasks)} videos to .mp4 (originals moved to {originals_dir})...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(convert_to_mp4, tasks), total=len(tasks)):
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    normalize_to_mp4_multiprocessing(args.input_dir, args.num_workers)
