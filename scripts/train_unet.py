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
import argparse
import re
import shutil
import datetime
import logging
from pathlib import Path
from omegaconf import OmegaConf

from tqdm.auto import tqdm
from einops import rearrange

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils.logging import get_logger
from diffusers.optimization import get_scheduler
from accelerate.utils import set_seed

from latentsync.data.unet_dataset import UNetDataset
from latentsync.models.unet import UNet3DConditionModel
from latentsync.models.stable_syncnet import StableSyncNet
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils.util import (
    cosine_loss,
    one_step_sampling,
    read_audio,
)
from latentsync.utils.audio import melspectrogram
from latentsync.whisper.audio2feature import Audio2Feature
from latentsync.trepa.loss import TREPALoss
import lpips


logger = get_logger(__name__)

# Sentinel for ckpt.resume_ckpt_path: pick up whatever the previous stage last wrote.
AUTO_RESUME = "auto"


def find_latest_checkpoint(train_output_dir: str):
    """Newest `checkpoint-*.pt` under `<train_output_dir>/<run>/checkpoints/`, or None.

    Run folders are named after their start time, so ordering by (folder name, step) is
    deterministic and does not depend on file timestamps.
    """
    checkpoints = []
    for checkpoint_path in Path(train_output_dir).glob("*/checkpoints/checkpoint-*.pt"):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_path.stem)
        if match:
            run_name = checkpoint_path.parent.parent.name
            checkpoints.append((run_name, int(match.group(1)), checkpoint_path))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda item: (item[0], item[1]))[2]


def training_state_path(checkpoint_path):
    """Sidecar holding optimizer / scheduler / scaler state for `checkpoint_path`.

    Kept out of the checkpoint itself so the weights file stays loadable by inference and stays the
    size it is now -- the 8-bit AdamW moments alone would add about 2.5 GiB to every save.
    """
    return str(Path(checkpoint_path).with_suffix(".training_state.pt"))


def save_training_state(checkpoint_path, optimizer, lr_scheduler, scaler):
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
        },
        training_state_path(checkpoint_path),
    )


def load_training_state(checkpoint_path, optimizer, lr_scheduler, scaler):
    """Restore optimizer/scheduler/scaler for a genuine same-stage resume. Returns True on success.

    Without this a resumed run keeps its step counter but restarts Adam from zeroed moments, which
    shows up as a loss spike, and replays the LR schedule from step 0 -- harmless for `constant`,
    wrong for anything with warmup or decay.
    """
    state_path = training_state_path(checkpoint_path)
    if not os.path.isfile(state_path):
        return False

    # A crash partway through save_training_state leaves a truncated sidecar, and editing
    # trainable_modules between runs makes the saved optimizer state no longer match. Neither is a
    # reason to refuse to train: warn, and carry on with fresh optimizer state.
    try:
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
    except Exception as e:
        logger.warning(f"Ignoring unusable training state {state_path}: {type(e).__name__} - {e}")
        return False
    return True


def prune_checkpoints(checkpoints_dir, max_keep):
    """Keep only the newest `max_keep` checkpoints, deleting their sidecars along with them.

    A full run writes max_train_steps / save_ckpt_steps checkpoints at roughly 7.5 GB each (5 GB of
    weights plus the training state), which is hundreds of GB on a real dataset. `max_keep` of 0 or
    less keeps everything.
    """
    if not max_keep or max_keep <= 0:
        return []

    checkpoints = []
    for checkpoint_path in Path(checkpoints_dir).glob("checkpoint-*.pt"):
        match = re.fullmatch(r"checkpoint-(\d+)", checkpoint_path.stem)
        if match:
            checkpoints.append((int(match.group(1)), checkpoint_path))

    removed = []
    for _, checkpoint_path in sorted(checkpoints)[:-max_keep]:
        for path in (checkpoint_path, Path(training_state_path(str(checkpoint_path)))):
            if path.is_file():
                path.unlink()
                removed.append(path.name)
    return removed


def build_optimizer(config, trainable_params):
    """AdamW, optionally keeping its moments in 8 bits.

    AdamW stores two fp32 moments per parameter. For this 1.27B UNet that is 10.1 GiB on its own,
    more than the weights and the gradients put together. bitsandbytes quantises the moments
    block-wise -- a separate fp32 scale per block of values -- which brings that down to about
    2.5 GiB. The per-block scale is what makes it safe: a plain fp16 moment would flush squared
    gradients (1e-8 and below) to zero, and a bf16 one has too few mantissa bits to hold them.
    """
    if not config.optimizer.get("use_8bit_adam", False):
        return torch.optim.AdamW(trainable_params, lr=config.optimizer.lr), "AdamW (fp32 states)"

    try:
        import bitsandbytes as bnb
    except ImportError as e:
        raise ImportError(
            "optimizer.use_8bit_adam is set, but bitsandbytes is not installed. Install it with "
            "`pip install bitsandbytes` -- on Blackwell (sm_120) you need a build against CUDA "
            "12.8 -- or set optimizer.use_8bit_adam to false."
        ) from e

    return bnb.optim.AdamW8bit(trainable_params, lr=config.optimizer.lr), "AdamW8bit (bitsandbytes)"


def resolve_resume_ckpt_path(config):
    """Resolve ckpt.resume_ckpt_path, returning (path, keep_global_step).

    "auto" resolves in two steps: this stage's own newest checkpoint under train_output_dir if
    there is one -- an interrupted run picking up where it stopped, so its step counter carries
    over -- otherwise the newest checkpoint under ckpt.resume_search_dir, which is the previous
    stage. In that second case the step counter has to be dropped: it belongs to the other stage,
    and carrying it over would put global_step past max_train_steps and the training loop would
    exit without doing anything.

    Anything other than "auto" is an explicit path, used as is with its stored step counter.
    """
    ckpt_path = config.ckpt.resume_ckpt_path
    if ckpt_path != AUTO_RESUME:
        return ckpt_path, True

    train_output_dir = config.data.train_output_dir
    own_checkpoint = find_latest_checkpoint(train_output_dir)
    if own_checkpoint is not None:
        return str(own_checkpoint), True

    search_dir = config.ckpt.get("resume_search_dir", None)
    if not search_dir:
        raise ValueError(
            f'ckpt.resume_ckpt_path is "{AUTO_RESUME}" but no checkpoint was found under '
            f"{train_output_dir}, and no ckpt.resume_search_dir is configured to fall back to. "
            "Set resume_search_dir to the previous stage's train_output_dir, or set "
            "resume_ckpt_path to an explicit checkpoint path."
        )

    previous_checkpoint = find_latest_checkpoint(search_dir)
    if previous_checkpoint is None:
        raise ValueError(
            f'ckpt.resume_ckpt_path is "{AUTO_RESUME}" but no checkpoint was found under '
            f"{train_output_dir} or {search_dir}. Run the previous stage first, or set "
            "resume_ckpt_path to an explicit checkpoint path."
        )
    return str(previous_checkpoint), False


@torch.no_grad()
def validation_sync_confidence(syncnet, generated_faces, audio_path, syncnet_config, device):
    """Score generated validation faces with the StableSyncNet used by training."""
    num_frames = syncnet_config.data.num_frames
    mel_window_length = math.ceil(num_frames / 5 * 16)
    audio_samples = read_audio(audio_path, syncnet_config.data.audio_sample_rate)
    original_mel = torch.from_numpy(melspectrogram(audio_samples.cpu().numpy()))
    confidences = []

    for start_idx in range(0, len(generated_faces) - num_frames + 1, num_frames):
        frames = generated_faces[start_idx : start_idx + num_frames]
        if frames.shape[-1] != syncnet_config.data.resolution:
            frames = F.interpolate(
                frames,
                size=(syncnet_config.data.resolution, syncnet_config.data.resolution),
                mode="bicubic",
            )
        frames = rearrange(frames, "f c h w -> 1 (f c) h w")
        if syncnet_config.data.lower_half:
            frames = frames[:, :, frames.shape[2] // 2 :, :]

        mel_start_idx = int(80.0 * (start_idx / float(syncnet_config.data.video_fps)))
        mel = original_mel[:, mel_start_idx : mel_start_idx + mel_window_length].unsqueeze(0)
        if mel.shape[-1] != mel_window_length:
            break

        vision_embeds, audio_embeds = syncnet(
            frames.to(device=device, dtype=torch.float16),
            mel.to(device=device, dtype=torch.float16),
        )
        confidence = F.cosine_similarity(vision_embeds.float(), audio_embeds.float()).mean()
        confidences.append(confidence.item())

    if not confidences:
        raise RuntimeError("Validation output is too short to calculate StableSyncNet confidence")
    return sum(confidences) / len(confidences)


def main(config):
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is available for training.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    set_seed(config.run.seed)

    # Logging folder
    # "%H-%M-%S" rather than "%H:%M:%S": a colon is not a legal filename character on
    # Windows, and os.makedirs below would fail with WinError 123.
    folder_name = "train" + datetime.datetime.now().strftime("-%Y_%m_%d-%H-%M-%S")
    output_dir = os.path.join(config.data.train_output_dir, folder_name)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    diffusers.utils.logging.set_verbosity_info()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{output_dir}/val_videos", exist_ok=True)
    shutil.copy(config.unet_config_path, output_dir)
    shutil.copy(config.data.syncnet_config_path, output_dir)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, "tensorboard"))

    noise_scheduler = DDIMScheduler.from_pretrained("configs")

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    vae.requires_grad_(False)
    vae.to(device)

    if config.run.pixel_space_supervise:
        vae.enable_gradient_checkpointing()

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        audio_embeds_cache_dir=config.data.audio_embeds_cache_dir,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    resume_ckpt_path, keep_global_step = resolve_resume_ckpt_path(config)
    continued = "resuming this stage" if keep_global_step else "initialising from the previous stage"
    logger.info(f"Resume checkpoint ({continued}): {resume_ckpt_path}")

    unet, resume_global_step = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        resume_ckpt_path,
        device=device,
    )
    if not keep_global_step:
        resume_global_step = 0

    if config.model.add_audio_layer and config.run.use_syncnet:
        syncnet_config = OmegaConf.load(config.data.syncnet_config_path)
        if syncnet_config.ckpt.inference_ckpt_path == "":
            raise ValueError("SyncNet path is not provided")
        syncnet = StableSyncNet(OmegaConf.to_container(syncnet_config.model), gradient_checkpointing=True).to(
            device=device, dtype=torch.float16
        )
        syncnet_checkpoint = torch.load(
            syncnet_config.ckpt.inference_ckpt_path, map_location=device, weights_only=True
        )
        syncnet.load_state_dict(syncnet_checkpoint["state_dict"])
        syncnet.requires_grad_(False)

        del syncnet_checkpoint
        torch.cuda.empty_cache()

    if config.model.use_motion_module:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            for trainable_module_name in config.run.trainable_modules:
                if trainable_module_name in name:
                    param.requires_grad = True
                    break
        trainable_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    else:
        unet.requires_grad_(True)
        trainable_params = list(unet.parameters())

    optimizer, optimizer_name = build_optimizer(config, trainable_params)

    logger.info(f"trainable params number: {len(trainable_params)}")
    logger.info(f"trainable params scale: {sum(p.numel() for p in trainable_params) / 1e6:.3f} M")
    logger.info(f"optimizer: {optimizer_name}")

    # Enable gradient checkpointing
    if config.run.enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Get the training dataset
    train_dataset = UNetDataset(config.data.train_data_dir, config)
    data_generator = torch.Generator()

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=False,
        drop_last=True,
        worker_init_fn=train_dataset.worker_init_fn,
        generator=data_generator,
    )

    # Get the training iteration
    if config.run.max_train_steps == -1:
        assert config.run.max_train_epochs != -1
        config.run.max_train_steps = config.run.max_train_epochs * len(train_dataloader)

    # Scheduler
    lr_scheduler = get_scheduler(
        config.optimizer.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.lr_warmup_steps,
        num_training_steps=config.run.max_train_steps,
    )

    if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
        lpips_loss_func = lpips.LPIPS(net="vgg").to(device)

    if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
        trepa_loss_func = TREPALoss(device=device, with_cp=True)

    # Validation pipeline
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=noise_scheduler,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(config.run.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running single-GPU training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Batch size = {config.data.batch_size}")
    logger.info(f"  Total optimization steps = {config.run.max_train_steps}")
    if resume_global_step >= config.run.max_train_steps:
        raise ValueError(
            f"Resuming at step {resume_global_step}, which is already at or past max_train_steps "
            f"({config.run.max_train_steps}). Training would exit without doing anything -- raise "
            "max_train_steps, or start from a different checkpoint."
        )
    global_step = resume_global_step
    first_epoch = resume_global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, config.run.max_train_steps),
        initial=resume_global_step,
        desc="Steps",
    )

    # Support mixed-precision training
    scaler = torch.amp.GradScaler("cuda") if config.run.mixed_precision_training else None

    # Only for a genuine same-stage resume. Starting a new stage reuses the previous stage's weights
    # but has a different trainable parameter set, so its optimizer state does not apply.
    if keep_global_step and resume_global_step > 0:
        if load_training_state(resume_ckpt_path, optimizer, lr_scheduler, scaler):
            logger.info(f"Restored optimizer/scheduler state from {training_state_path(resume_ckpt_path)}")
        else:
            logger.warning(
                f"No training state next to {resume_ckpt_path}; resuming with zeroed optimizer "
                "moments and the LR schedule replayed from step 0."
            )

    for epoch in range(first_epoch, num_train_epochs):
        # Keep each epoch's shuffle deterministic, including after resuming from a checkpoint.
        data_generator.manual_seed(config.run.seed + epoch)
        unet.train()

        for step, batch in enumerate(train_dataloader):
            ### >>>> Training >>>> ###

            if config.model.add_audio_layer:
                if batch["mel"] != []:
                    mel = batch["mel"].to(device, dtype=torch.float16)

                audio_embeds_list = []
                try:
                    for idx in range(len(batch["video_path"])):
                        video_path = batch["video_path"][idx]
                        start_idx = batch["start_idx"][idx]

                        with torch.no_grad():
                            audio_feat = audio_encoder.audio2feat(video_path)
                        audio_embeds = audio_encoder.crop_overlap_audio_window(audio_feat, start_idx)
                        audio_embeds_list.append(audio_embeds)
                except Exception as e:
                    logger.info(f"{type(e).__name__} - {e} - {video_path}")
                    continue
                audio_embeds = torch.stack(audio_embeds_list)  # (B, 16, 50, 384)
                audio_embeds = audio_embeds.to(device, dtype=torch.float16)
            else:
                audio_embeds = None

            # Convert videos to latent space
            gt_pixel_values = batch["gt_pixel_values"].to(device, dtype=torch.float16)
            masked_pixel_values = batch["masked_pixel_values"].to(device, dtype=torch.float16)
            masks = batch["masks"].to(device, dtype=torch.float16)
            ref_pixel_values = batch["ref_pixel_values"].to(device, dtype=torch.float16)

            gt_pixel_values = rearrange(gt_pixel_values, "b f c h w -> (b f) c h w")
            masked_pixel_values = rearrange(masked_pixel_values, "b f c h w -> (b f) c h w")
            masks = rearrange(masks, "b f c h w -> (b f) c h w")
            ref_pixel_values = rearrange(ref_pixel_values, "b f c h w -> (b f) c h w")

            with torch.no_grad():
                gt_latents = vae.encode(gt_pixel_values).latent_dist.sample()
                masked_latents = vae.encode(masked_pixel_values).latent_dist.sample()
                ref_latents = vae.encode(ref_pixel_values).latent_dist.sample()

            masks = torch.nn.functional.interpolate(masks, size=config.data.resolution // vae_scale_factor)

            gt_latents = (
                rearrange(gt_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            masked_latents = (
                rearrange(masked_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames)
                - vae.config.shift_factor
            ) * vae.config.scaling_factor
            ref_latents = (
                rearrange(ref_latents, "(b f) c h w -> b c f h w", f=config.data.num_frames) - vae.config.shift_factor
            ) * vae.config.scaling_factor
            masks = rearrange(masks, "(b f) c h w -> b c f h w", f=config.data.num_frames)

            # Sample noise that we'll add to the latents
            if config.run.use_mixed_noise:
                # Refer to the paper: https://arxiv.org/abs/2305.10474
                noise_shared_std_dev = (config.run.mixed_noise_alpha**2 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_shared = torch.randn_like(gt_latents) * noise_shared_std_dev
                noise_shared = noise_shared[:, :, 0:1].repeat(1, 1, config.data.num_frames, 1, 1)

                noise_ind_std_dev = (1 / (1 + config.run.mixed_noise_alpha**2)) ** 0.5
                noise_ind = torch.randn_like(gt_latents) * noise_ind_std_dev
                noise = noise_ind + noise_shared
            else:
                noise = torch.randn_like(gt_latents)
                noise = noise[:, :, 0:1].repeat(
                    1, 1, config.data.num_frames, 1, 1
                )  # Using the same noise for all frames, refer to the paper: https://arxiv.org/abs/2308.09716

            bsz = gt_latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=gt_latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_gt_latents = noise_scheduler.add_noise(gt_latents, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                raise NotImplementedError
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            unet_input = torch.cat([noisy_gt_latents, masks, masked_latents, ref_latents], dim=1)

            # Predict the noise and compute loss
            # Mixed-precision training
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=config.run.mixed_precision_training):
                pred_noise = unet(unet_input, timesteps, encoder_hidden_states=audio_embeds).sample

            if config.run.recon_loss_weight != 0:
                recon_loss = F.mse_loss(pred_noise.float(), target.float(), reduction="mean")
            else:
                recon_loss = 0

            pred_latents = one_step_sampling(noise_scheduler, pred_noise, timesteps, noisy_gt_latents)

            if config.run.pixel_space_supervise:
                pred_pixel_values = vae.decode(
                    rearrange(pred_latents, "b c f h w -> (b f) c h w") / vae.config.scaling_factor
                    + vae.config.shift_factor
                ).sample

            if config.run.perceptual_loss_weight != 0 and config.run.pixel_space_supervise:
                pred_pixel_values_perceptual = pred_pixel_values[:, :, pred_pixel_values.shape[2] // 2 :, :]
                gt_pixel_values_perceptual = gt_pixel_values[:, :, gt_pixel_values.shape[2] // 2 :, :]
                lpips_loss = lpips_loss_func(
                    pred_pixel_values_perceptual.float(), gt_pixel_values_perceptual.float()
                ).mean()
            else:
                lpips_loss = 0

            if config.run.trepa_loss_weight != 0 and config.run.pixel_space_supervise:
                trepa_pred_pixel_values = rearrange(
                    pred_pixel_values, "(b f) c h w -> b c f h w", f=config.data.num_frames
                )
                trepa_gt_pixel_values = rearrange(
                    gt_pixel_values, "(b f) c h w -> b c f h w", f=config.data.num_frames
                )
                trepa_loss = trepa_loss_func(trepa_pred_pixel_values, trepa_gt_pixel_values)
            else:
                trepa_loss = 0

            if config.model.add_audio_layer and config.run.use_syncnet:
                if config.run.pixel_space_supervise:
                    if config.data.resolution != syncnet_config.data.resolution:
                        pred_pixel_values = F.interpolate(
                            pred_pixel_values,
                            size=(syncnet_config.data.resolution, syncnet_config.data.resolution),
                            mode="bicubic",
                        )
                    syncnet_input = rearrange(
                        pred_pixel_values, "(b f) c h w -> b (f c) h w", f=config.data.num_frames
                    )
                else:
                    syncnet_input = rearrange(pred_latents, "b c f h w -> b (f c) h w")

                if syncnet_config.data.lower_half:
                    height = syncnet_input.shape[2]
                    syncnet_input = syncnet_input[:, :, height // 2 :, :]
                ones_tensor = torch.ones((config.data.batch_size, 1)).float().to(device=device)
                vision_embeds, audio_embeds = syncnet(syncnet_input, mel)
                sync_loss = cosine_loss(vision_embeds.float(), audio_embeds.float(), ones_tensor).mean()
            else:
                sync_loss = 0

            loss = (
                recon_loss * config.run.recon_loss_weight
                + sync_loss * config.run.sync_loss_weight
                + lpips_loss * config.run.perceptual_loss_weight
                + trepa_loss * config.run.trepa_loss_weight
            )

            optimizer.zero_grad()

            # Backpropagate
            if config.run.mixed_precision_training:
                scaler.scale(loss).backward()
                """ >>> gradient clipping >>> """
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, config.optimizer.max_grad_norm)
                """ <<< gradient clipping <<< """
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                """ >>> gradient clipping >>> """
                torch.nn.utils.clip_grad_norm_(trainable_params, config.optimizer.max_grad_norm)
                """ <<< gradient clipping <<< """
                optimizer.step()

            # Check the grad of attn blocks for debugging
            # print(unet.up_blocks[3].attentions[2].transformer_blocks[0].attn2.to_q.weight.grad)

            lr_scheduler.step()
            progress_bar.update(1)
            global_step += 1

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/reconstruction_loss", float(recon_loss), global_step)
            writer.add_scalar("train/sync_loss", float(sync_loss), global_step)
            writer.add_scalar("train/perceptual_loss", float(lpips_loss), global_step)
            writer.add_scalar("train/trepa_loss", float(trepa_loss), global_step)
            writer.add_scalar("train/learning_rate", lr_scheduler.get_last_lr()[0], global_step)

            ### <<<< Training <<<< ###

            # Save checkpoint and conduct validation
            if global_step % config.ckpt.save_ckpt_steps == 0:
                model_save_path = os.path.join(output_dir, f"checkpoints/checkpoint-{global_step}.pt")
                state_dict = {
                    "global_step": global_step,
                    "state_dict": unet.state_dict(),
                }
                try:
                    torch.save(state_dict, model_save_path)
                    logger.info(f"Saved checkpoint to {model_save_path}")
                except Exception as e:
                    logger.error(f"Error saving model: {e}")

                # Separate from the weights save: the sidecar is only needed to resume, so losing
                # it (a full disk, most likely) must not be reported as having lost the checkpoint.
                try:
                    save_training_state(model_save_path, optimizer, lr_scheduler, scaler)
                except Exception as e:
                    logger.error(f"Error saving training state (checkpoint itself is fine): {e}")

                try:
                    removed = prune_checkpoints(
                        os.path.dirname(model_save_path), config.ckpt.get("max_keep_ckpts", 0)
                    )
                    if removed:
                        logger.info(f"Pruned {len(removed)} old checkpoint file(s): {', '.join(removed)}")
                except Exception as e:
                    logger.error(f"Error pruning old checkpoints: {e}")

                # Validation
                logger.info("Running validation... ")

                validation_video_out_path = os.path.join(output_dir, f"val_videos/val_video_{global_step}.mp4")

                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    generated_faces = pipeline(
                        config.data.val_video_path,
                        config.data.val_audio_path,
                        validation_video_out_path,
                        num_frames=config.data.num_frames,
                        num_inference_steps=config.run.inference_steps,
                        guidance_scale=config.run.guidance_scale,
                        weight_dtype=torch.float16,
                        width=config.data.resolution,
                        height=config.data.resolution,
                        mask_image_path=config.data.mask_image_path,
                        return_generated_faces=True,
                    )

                logger.info(f"Saved validation video output to {validation_video_out_path}")

                # Peak includes the validation pipeline above, so it is the true high-water mark.
                peak_gib = torch.cuda.max_memory_allocated(device) / 2**30
                writer.add_scalar("train/peak_vram_gib", peak_gib, global_step)
                logger.info(f"Peak VRAM so far: {peak_gib:.1f} GiB")
                torch.cuda.reset_peak_memory_stats(device)

                if config.model.add_audio_layer and config.run.use_syncnet:
                    try:
                        conf = validation_sync_confidence(
                            syncnet,
                            generated_faces,
                            config.data.val_audio_path,
                            syncnet_config,
                            device,
                        )
                        writer.add_scalar("validation/sync_confidence", conf, global_step)
                        logger.info(f"Validation StableSyncNet confidence at step {global_step}: {conf:.4f}")
                    except Exception as e:
                        logger.warning(f"Unable to calculate validation sync confidence: {type(e).__name__} - {e}")

                writer.flush()

            logs = {"step_loss": loss.item(), "epoch": epoch}
            progress_bar.set_postfix(**logs)

            if global_step >= config.run.max_train_steps:
                break

    progress_bar.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config file path
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")

    args = parser.parse_args()
    config = OmegaConf.load(args.unet_config_path)
    config.unet_config_path = args.unet_config_path

    main(config)
