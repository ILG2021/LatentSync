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


from latentsync.utils.util import check_ffmpeg_installed


def gather_mov_paths_recursively(input_dir):
    print(f"Recursively gathering mov paths of {input_dir} ...")
    paths = []
    gather_mov_paths(input_dir, paths)
    return paths


def gather_mov_paths(input_dir, paths):
    for file in sorted(os.listdir(input_dir)):
        if file.lower().endswith(".mov"):
            filepath = os.path.join(input_dir, file)
            paths.append(filepath)
        elif os.path.isdir(os.path.join(input_dir, file)):
            gather_mov_paths(os.path.join(input_dir, file), paths)


def convert_mov_to_mp4(mov_path):
    mp4_path = mov_path.rsplit(".", 1)[0] + ".mp4"
    # Using ffmpeg to convert mov to mp4 with high quality settings
    # -y: overwrite output files without asking
    # -c:v libx264 -crf 17: visually lossless quality
    # -preset slow: better quality/size ratio
    # -c:a aac -b:a 192k: high quality audio
    # -loglevel error: only show errors
    command = f'ffmpeg -y -i "{mov_path}" -c:v libx264 -crf 18 -preset slow -c:a aac -b:a 192k -strict experimental -loglevel info "{mp4_path}"'
    try:
        subprocess.run(command, shell=True, check=True)
        # If successful, remove the original mov file
        os.remove(mov_path)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {mov_path}: {e}")


def convert_mov_to_mp4_multiprocessing(input_dir, num_workers):
    check_ffmpeg_installed()
    mov_paths = gather_mov_paths_recursively(input_dir)
    if not mov_paths:
        print("No .mov files found.")
        return

    print(f"Converting {len(mov_paths)} .mov files to .mp4...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(convert_mov_to_mp4, mov_paths), total=len(mov_paths)):
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=10)
    args = parser.parse_args()

    convert_mov_to_mp4_multiprocessing(args.input_dir, args.num_workers)
