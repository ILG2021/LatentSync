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
import math
import os
import random
import re

import yaml
from tqdm import tqdm

from latentsync.utils.util import gather_video_paths_recursively

# `segment_videos` names its output "<source video>_%03d.mp4", so stripping the trailing index
# recovers which source recording a segment came from. Only used for reporting.
SEGMENT_INDEX_PATTERN = re.compile(r"_\d+$")

# The configs to keep in sync with the dataset, and how many times each segment should be visited
# over that stage's run. UNetDataset draws a random 16-frame window per visit, so a segment is not
# exhausted the way a fixed sample would be, but past roughly this many passes the stage starts
# memorising instead of adapting.
#
# stage2 gets fewer passes than stage1: its backbone is frozen so there is far less to fit, while
# each step costs a VAE decode plus the SyncNet, LPIPS and TREPA losses. Treat both numbers as a
# starting budget -- stage2 in particular is usually stopped by watching the validation video and
# validation/sync_confidence rather than by running the budget out.
UNET_CONFIGS = {
    "configs/unet/stage1_512.yaml": 30,
    "configs/unet/stage2_512.yaml": 15,
    # Fast single-5090 preset: use a short starting budget and stop on validation quality.
    "configs/unet/stage2_512_full_5090_offload.yaml": 3,
}
# Small datasets still need enough optimizer steps for the domain shift to take hold.
MIN_TRAIN_STEPS = 2000
STEP_ROUNDING = 500


def source_video_of(video_path: str) -> str:
    stem = os.path.splitext(os.path.basename(video_path))[0]
    return SEGMENT_INDEX_PATTERN.sub("", stem)


def load_pinned_val_clips(path):
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def save_pinned_val_clips(path, video_paths):
    with open(path, "w", encoding="utf-8") as f:
        for video_path in video_paths:
            f.write(video_path + "\n")


def pick_validation_clips(all_video_paths, num_val_clips, pinned_clips):
    """Hold out `num_val_clips` random segments; everything else goes to training.

    The chosen clips are pinned to a file so that re-running after adding footage keeps the same
    split -- otherwise a re-run would move the old validation clips into training and make
    validation numbers from before and after incomparable. Clips that have disappeared from the
    dataset are replaced, keeping the rest.
    """
    by_abspath = {os.path.abspath(video_path): video_path for video_path in all_video_paths}

    held_out = []
    for pinned in pinned_clips:
        video_path = by_abspath.get(os.path.abspath(pinned))
        if video_path is not None and video_path not in held_out:
            held_out.append(video_path)
    if len(pinned_clips) > len(held_out):
        print(f"Warning: {len(pinned_clips) - len(held_out)} pinned validation clips are gone; topping up.")

    # Always leave something to train on.
    target = min(num_val_clips, max(0, len(all_video_paths) - 1))
    if target < num_val_clips:
        print(f"Warning: only {len(all_video_paths)} segments available, holding out {target} instead of {num_val_clips}.")

    held_out = held_out[:target]
    if len(held_out) < target:
        pool = sorted(set(all_video_paths) - set(held_out))
        held_out += random.sample(pool, target - len(held_out))

    return sorted(held_out)


def compute_max_train_steps(num_segments, batch_size, passes_per_segment, num_processes=1):
    """Optimizer steps needed for each segment to be seen `passes_per_segment` times.

    One optimizer step consumes batch_size * num_processes samples, so the step count has to be
    divided by the effective batch. Pass num_processes when training on more than one GPU.
    """
    effective_batch = max(1, batch_size * num_processes)
    steps = math.ceil(num_segments * passes_per_segment / effective_batch)
    steps = math.ceil(steps / STEP_ROUNDING) * STEP_ROUNDING
    return max(MIN_TRAIN_STEPS, steps)


def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def yaml_quote(video_path: str) -> str:
    # Single-quoted YAML has no escape sequences, so Windows backslashes survive intact. We still
    # normalise to forward slashes because ffmpeg and decord accept them on every platform.
    return "'" + video_path.replace(os.sep, "/").replace("'", "''") + "'"


def update_unet_config(config_path, values):
    """Rewrite just the given top-level-indented keys, leaving the rest of the file untouched.

    Round-tripping through yaml.safe_load/yaml.dump would silently strip every comment in the
    config, and these files carry the tuning notes (`# [1.0 - 3.0]`, `# 49`, ...) that make them
    readable.
    """
    # newline="" keeps each line terminator as it is on disk, so rewriting a few lines does not
    # convert the whole file between CRLF and LF and blow up the diff.
    with open(config_path, "r", encoding="utf-8", newline="") as f:
        lines = f.readlines()

    replaced = set()
    for i, line in enumerate(lines):
        for key, value in values.items():
            if line.strip().startswith(f"{key}:"):
                prefix = line[: len(line) - len(line.lstrip())]
                eol = "\r\n" if line.endswith("\r\n") else "\n"
                lines[i] = f"{prefix}{key}: {value}{eol}"
                replaced.add(key)

    missing = set(values) - replaced
    if missing:
        raise SystemExit(f"Could not find {sorted(missing)} in {config_path}; update it by hand.")

    with open(config_path, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="my_data/affine_transformed")
    parser.add_argument("--fileslist_path", type=str, default="my_data/fileslist.txt")
    parser.add_argument("--val_clips_path", type=str, default="my_data/val_clips.txt")
    parser.add_argument("--num_val_clips", type=int, default=10)
    # Must match --nproc_per_node in train_unet.sh: one optimizer step consumes
    # batch_size * num_processes samples, so a stale value here scales max_train_steps wrongly.
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument(
        "--skip_config_update", action="store_true",
        help="Only rewrite dataset lists; preserve validation paths and training budgets in configs.",
    )
    args = parser.parse_args()
    if args.num_val_clips < 1:
        parser.error("--num_val_clips must be at least 1")

    random.seed(args.seed)

    print(f"Gathering videos from: {args.dataset_dir}")
    all_video_paths = gather_video_paths_recursively(args.dataset_dir)

    if not all_video_paths:
        raise SystemExit(f"No video segments found in {args.dataset_dir}. Run the data processing pipeline first.")

    val_video_paths = pick_validation_clips(
        all_video_paths, args.num_val_clips, load_pinned_val_clips(args.val_clips_path)
    )
    held_out = set(os.path.abspath(video_path) for video_path in val_video_paths)

    train_video_paths = [
        video_path for video_path in all_video_paths if os.path.abspath(video_path) not in held_out
    ]
    if not train_video_paths:
        raise SystemExit("Holding out the validation video left no training data.")

    save_pinned_val_clips(args.val_clips_path, val_video_paths)

    with open(args.fileslist_path, "w", encoding="utf-8") as f:
        for video_path in tqdm(train_video_paths):
            f.write(f"{video_path}\n")

    print(
        f"\nWrote {len(train_video_paths)} training segments to {args.fileslist_path} "
        f"({len(val_video_paths)} clips from {len(set(source_video_of(p) for p in val_video_paths))} "
        f"source recording(s) held out for validation)."
    )

    config_updates = {} if args.skip_config_update else UNET_CONFIGS
    for unet_config_path, passes_per_segment in config_updates.items():
        if not os.path.isfile(unet_config_path):
            print(f"Skipping missing config: {unet_config_path}")
            continue
        config = read_config(unet_config_path)
        batch_size = config["data"]["batch_size"]
        save_ckpt_steps = config["ckpt"]["save_ckpt_steps"]
        max_train_steps = compute_max_train_steps(
            len(train_video_paths), batch_size, passes_per_segment, args.num_processes
        )

        update_unet_config(
            unet_config_path,
            {
                "val_video_path": yaml_quote(val_video_paths[0]),
                "val_audio_path": yaml_quote(val_video_paths[0]),
                "max_train_steps": max_train_steps,
            },
        )
        print(
            f"  {unet_config_path}: max_train_steps={max_train_steps} "
            f"({passes_per_segment} passes over {len(train_video_paths)} segments, "
            f"batch {batch_size} x {args.num_processes}) "
            f"-> {math.ceil(max_train_steps / save_ckpt_steps)} checkpoints at "
            f"save_ckpt_steps={save_ckpt_steps}"
        )
