"""Inspect UNet checkpoints and optionally trace one LatentSync inference run.

The checkpoint scan runs on CPU and reports non-finite tensors plus unusually
large parameter scales.  The inference trace reports the UNet prediction,
DDIM latent, VAE decoder output, and final generated face tensors so a black
output can be localized without changing the training or inference pipeline.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

# Running ``python tools/diagnose_unet_checkpoint.py`` puts ``tools`` rather
# than the repository root on sys.path. Resolve imports relative to this file
# so the command works regardless of the caller's current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf

from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.whisper.audio2feature import Audio2Feature


def tensor_stats(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    finite = torch.isfinite(value)
    finite_count = int(finite.sum().item())
    total = value.numel()
    nan_count = int(torch.isnan(value).sum().item())
    posinf_count = int(torch.isposinf(value).sum().item())
    neginf_count = int(torch.isneginf(value).sum().item())

    result: dict[str, Any] = {
        "name": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": total,
        "finite": finite_count,
        "nan": nan_count,
        "posinf": posinf_count,
        "neginf": neginf_count,
    }
    if finite_count:
        valid = value[finite].float()
        result.update(
            minimum=float(valid.min().item()),
            maximum=float(valid.max().item()),
            mean=float(valid.mean().item()),
            std=float(valid.std(unbiased=False).item()),
            max_abs=float(valid.abs().max().item()),
            l2=float(torch.linalg.vector_norm(valid).item()),
        )
    return result


def print_tensor_stats(stats: dict[str, Any], prefix: str = "[TENSOR]") -> None:
    counts = (
        f"finite={stats['finite']}/{stats['numel']}, nan={stats['nan']}, "
        f"+inf={stats['posinf']}, -inf={stats['neginf']}"
    )
    if stats["finite"]:
        values = (
            f"min={stats['minimum']:.6g}, max={stats['maximum']:.6g}, "
            f"mean={stats['mean']:.6g}, std={stats['std']:.6g}, "
            f"max_abs={stats['max_abs']:.6g}"
        )
    else:
        values = "no finite values"
    print(
        f"{prefix} {stats['name']}: shape={tuple(stats['shape'])}, "
        f"dtype={stats['dtype']}, {counts}, {values}"
    )


def load_checkpoint_state(path: Path) -> tuple[int | None, dict[str, torch.Tensor]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path} does not contain a state_dict")
    state = checkpoint["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"{path} state_dict is not a mapping")
    return checkpoint.get("global_step"), state


def scan_checkpoint(path: Path) -> dict[str, Any]:
    print(f"\n===== Scanning checkpoint: {path} =====")
    global_step, state = load_checkpoint_state(path)
    parameter_stats = []
    bad = []

    for name, tensor in state.items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        stats = tensor_stats(name, tensor)
        parameter_stats.append(stats)
        if stats["finite"] != stats["numel"]:
            bad.append(stats)

    largest = sorted(parameter_stats, key=lambda item: item.get("max_abs", -1.0), reverse=True)[:20]
    print(f"global_step: {global_step}")
    print(f"floating tensors: {len(parameter_stats)}")
    print(f"tensors containing NaN/Inf: {len(bad)}")
    for stats in bad[:50]:
        print_tensor_stats(stats, prefix="[BAD]")
    print("Largest 20 tensors by absolute value:")
    for stats in largest:
        print(f"  {stats['max_abs']:.6g}  {stats['name']}  {tuple(stats['shape'])}")

    summary = {
        "path": str(path.resolve()),
        "global_step": global_step,
        "floating_tensor_count": len(parameter_stats),
        "bad_tensor_count": len(bad),
        "bad_tensors": bad,
        "largest_tensors": largest,
        "parameter_stats": parameter_stats,
    }
    del state
    gc.collect()
    return summary


def compare_checkpoints(reference_path: Path, candidate_path: Path) -> list[dict[str, Any]]:
    print(f"\n===== Comparing weights: {reference_path.name} -> {candidate_path.name} =====")
    _, reference = load_checkpoint_state(reference_path)
    _, candidate = load_checkpoint_state(candidate_path)
    changes = []
    for name, after in candidate.items():
        before = reference.get(name)
        if (
            before is None
            or not torch.is_tensor(before)
            or not torch.is_tensor(after)
            or not before.is_floating_point()
            or before.shape != after.shape
        ):
            continue
        before_float = before.float()
        after_float = after.float()
        delta = after_float - before_float
        reference_l2 = float(torch.linalg.vector_norm(before_float).item())
        candidate_l2 = float(torch.linalg.vector_norm(after_float).item())
        delta_l2 = float(torch.linalg.vector_norm(delta).item())
        changes.append(
            {
                "name": name,
                "relative_delta_l2": delta_l2 / max(reference_l2, 1e-30),
                "delta_l2": delta_l2,
                "max_abs_delta": float(delta.abs().max().item()),
                "reference_l2": reference_l2,
                "candidate_l2": candidate_l2,
            }
        )

    changes.sort(key=lambda item: item["relative_delta_l2"], reverse=True)
    print("Largest relative parameter changes:")
    for item in changes[:30]:
        print(
            f"  relative_l2={item['relative_delta_l2']:.6g}  "
            f"max_abs_delta={item['max_abs_delta']:.6g}  "
            f"{item['name']}"
        )
    del reference, candidate
    gc.collect()
    return changes


def extract_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    sample = getattr(output, "sample", None)
    if torch.is_tensor(sample):
        return sample
    if isinstance(output, (tuple, list)):
        for value in output:
            tensor = extract_tensor(value)
            if tensor is not None:
                return tensor
    return None


class InferenceTrace:
    def __init__(self, inference_steps: int, fail_on_nonfinite: bool):
        self.inference_steps = inference_steps
        self.fail_on_nonfinite = fail_on_nonfinite
        self.unet_calls = 0
        self.vae_calls = 0
        self.ddim_calls = 0
        self.records: list[dict[str, Any]] = []

    def record(self, name: str, tensor: torch.Tensor) -> None:
        stats = tensor_stats(name, tensor)
        self.records.append(stats)
        print_tensor_stats(stats)
        if self.fail_on_nonfinite and stats["finite"] != stats["numel"]:
            raise FloatingPointError(f"{name} contains NaN or Inf")

    def unet_hook(self, _module, _inputs, output) -> None:
        call = self.unet_calls
        self.unet_calls += 1
        # Trace the first video chunk only. Later chunks use the same denoising path.
        if call >= self.inference_steps:
            return
        tensor = extract_tensor(output)
        if tensor is not None:
            self.record(f"unet_noise_pred/call_{call}", tensor)

    def vae_hook(self, _module, _inputs, output) -> None:
        tensor = extract_tensor(output)
        if tensor is not None:
            self.record(f"vae_decoder_output/call_{self.vae_calls}", tensor)
        self.vae_calls += 1

    def ddim_callback(self, step: int, timestep: int, latents: torch.Tensor) -> None:
        if self.ddim_calls >= self.inference_steps:
            return
        self.ddim_calls += 1
        self.record(f"ddim_latents/step_{step}_t_{int(timestep)}", latents)


def run_inference_trace(args, config) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the inference trace")
    if not args.video_path or not args.audio_path:
        raise ValueError("--video-path and --audio-path are required with --trace-checkpoint")
    if not Path(args.video_path).is_file():
        raise FileNotFoundError(args.video_path)
    if not Path(args.audio_path).is_file():
        raise FileNotFoundError(args.audio_path)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device("cuda:0")
    set_seed(args.seed)

    scheduler = DDIMScheduler.from_pretrained("configs")
    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise ValueError(f"Unsupported cross_attention_dim: {config.model.cross_attention_dim}")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device=device,
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model), args.trace_checkpoint, device="cpu"
    )
    unet = unet.to(dtype=dtype)
    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to(device)

    if args.enable_deepcache:
        from DeepCache import DeepCacheSDHelper

        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()

    output_path = Path(args.video_out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.temp_dir).mkdir(parents=True, exist_ok=True)

    trace = InferenceTrace(args.inference_steps, args.fail_on_nonfinite)
    unet_handle = unet.register_forward_hook(trace.unet_hook)
    vae_handle = vae.decoder.register_forward_hook(trace.vae_hook)
    try:
        generated_faces = pipeline(
            video_path=args.video_path,
            audio_path=args.audio_path,
            video_out_path=str(output_path),
            start_time=args.start_time,
            num_frames=config.data.num_frames,
            num_inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=args.temp_dir,
            callback=trace.ddim_callback,
            callback_steps=1,
            return_generated_faces=True,
        )
    finally:
        unet_handle.remove()
        vae_handle.remove()

    trace.record("generated_faces", generated_faces)
    faces = generated_faces.detach().float()
    black_pixel_ratio = float((faces.amax(dim=1) <= -0.90).float().mean().item())
    clipped_black_ratio = float((faces <= -0.95).float().mean().item())
    print(f"[RESULT] black pixel ratio (all RGB <= -0.90): {black_pixel_ratio:.4%}")
    print(f"[RESULT] channel values <= -0.95: {clipped_black_ratio:.4%}")
    print(f"[RESULT] output video: {output_path.resolve()}")
    return {
        "checkpoint": str(Path(args.trace_checkpoint).resolve()),
        "dtype": args.dtype,
        "seed": args.seed,
        "guidance_scale": args.guidance_scale,
        "inference_steps": args.inference_steps,
        "deepcache": args.enable_deepcache,
        "black_pixel_ratio": black_pixel_ratio,
        "clipped_black_channel_ratio": clipped_black_ratio,
        "output_video": str(output_path.resolve()),
        "tensors": trace.records,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scan UNet checkpoints and optionally trace an inference that produces black faces."
    )
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint files to scan")
    parser.add_argument("--unet-config-path", default="configs/unet/stage2_512_full_5090_offload.yaml")
    parser.add_argument("--trace-checkpoint", help="Checkpoint to use for the optional inference trace")
    parser.add_argument("--video-path")
    parser.add_argument("--audio-path")
    parser.add_argument("--video-out-path", default="debug/diagnose_unet/diagnostic_output.mp4")
    parser.add_argument("--temp-dir", default="temp/diagnose_unet")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--enable-deepcache", action="store_true")
    parser.add_argument("--fail-on-nonfinite", action="store_true")
    parser.add_argument("--report-json", default="debug/diagnose_unet/report.json")
    args = parser.parse_args()
    if not math.isfinite(args.start_time) or args.start_time < 0:
        parser.error("--start-time must be finite and non-negative")
    if args.inference_steps <= 0:
        parser.error("--inference-steps must be positive")
    return args


def main() -> None:
    args = parse_args()
    checkpoint_paths = [Path(path) for path in args.checkpoints]
    for path in checkpoint_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    report: dict[str, Any] = {"checkpoint_scans": []}
    for path in checkpoint_paths:
        report["checkpoint_scans"].append(scan_checkpoint(path))

    if len(report["checkpoint_scans"]) > 1:
        report["weight_changes"] = compare_checkpoints(checkpoint_paths[0], checkpoint_paths[1])

    if args.trace_checkpoint:
        config = OmegaConf.load(args.unet_config_path)
        report["inference_trace"] = run_inference_trace(args, config)

    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"\nDiagnostic report written to: {report_path.resolve()}")


if __name__ == "__main__":
    main()
