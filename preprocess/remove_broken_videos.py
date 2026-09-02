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
from multiprocessing import Pool
import tqdm

from latentsync.utils.av_reader import AVReader
from latentsync.utils.util import gather_video_paths_recursively


def quarantine_broken_video(args):
    """Move a video that decord cannot open out of the dataset, instead of deleting it.

    AVReader builds the AudioReader first, so this also catches videos with no audio track. Those
    would otherwise survive `resample_fps_hz` (its `-map 0:a:0?` makes audio optional) and then kill
    an affine transform worker when combine_video_audio fails to extract the audio.

    We move rather than delete: `input_dir` holds the original footage, and a bare `except` also
    fires on transient failures such as running out of file handles under a large worker pool.
    """
    video_path, input_dir, quarantine_dir = args
    try:
        AVReader(video_path)
    except Exception as e:
        # Mirror the layout under input_dir instead of flattening to a basename: two clips with the
        # same name in different subdirectories would otherwise overwrite each other here.
        destination = os.path.join(quarantine_dir, os.path.relpath(video_path, input_dir))
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(video_path, destination)
        print(f"Quarantined: {video_path} - {type(e).__name__}: {e}")
        return video_path
    return None


def remove_broken_videos_multiprocessing(input_dir, num_workers, quarantine_dir=None):
    video_paths = gather_video_paths_recursively(input_dir)
    if quarantine_dir is None:
        quarantine_dir = os.path.join(os.path.dirname(os.path.abspath(input_dir)), "broken")
    tasks = [(video_path, input_dir, quarantine_dir) for video_path in video_paths]

    print("Quarantining broken videos...")
    with Pool(num_workers) as pool:
        results = list(tqdm.tqdm(pool.imap_unordered(quarantine_broken_video, tasks), total=len(tasks)))

    quarantined = [video_path for video_path in results if video_path is not None]
    if quarantined:
        print(f"Quarantined {len(quarantined)} broken videos to {quarantine_dir}")


if __name__ == "__main__":
    input_dir = "my_data/raw"
    num_workers = 8

    remove_broken_videos_multiprocessing(input_dir, num_workers)
