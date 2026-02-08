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

import yaml
import os
import random
from tqdm import tqdm
from latentsync.utils.util import gather_video_paths_recursively


class FileslistWriter:
    def __init__(self, fileslist_path: str):
        self.fileslist_path = fileslist_path
        with open(fileslist_path, "w") as _:
            pass

    def write_dataset(self, dataset_dir: str, exclude_path: str = None):
        print(f"Gathering videos from: {dataset_dir}")
        video_paths = gather_video_paths_recursively(dataset_dir)
        
        with open(self.fileslist_path, "w") as f:
            for video_path in tqdm(video_paths):
                if exclude_path and os.path.abspath(video_path) == os.path.abspath(exclude_path):
                    continue
                f.write(f"{video_path}\n")
        return video_paths


def update_unet_config(config_path, val_video_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if "data" not in config:
        config["data"] = {}

    config["data"]["val_video_path"] = val_video_path
    config["data"]["val_audio_path"] = val_video_path

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Successfully updated {config_path} with a random video as validation target.")


if __name__ == "__main__":
    fileslist_path = "my_data/fileslist.txt"
    unet_config_path = "configs/unet/stage1_512.yaml"
    dataset_dir = "high_visual_quality"

    # 1. 扫描所有视频获取候选列表
    all_video_paths = gather_video_paths_recursively(dataset_dir)

    if all_video_paths:
        # 2. 随机挑选一个作为验证集
        val_video = random.choice(all_video_paths)
        print(f"Randomly selected validation video: {val_video}")
        
        # 3. 更新配置文件
        update_unet_config(unet_config_path, val_video)
        
        # 4. 写入索引文件，同时剔除刚才选中的验证视频
        writer = FileslistWriter(fileslist_path)
        writer.write_dataset(dataset_dir, exclude_path=val_video)
        print(f"Finished writing {fileslist_path}, excluded the validation video.")
    else:
        print("Warning: No video segments found in high_visual_quality. Did the pre-processing run correctly?")
