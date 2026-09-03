"""Run all matching audio/video pairs in a folder and write a test report."""
import argparse
import csv
import json
import subprocess
import sys
import time
import wave
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}


def audio_duration_seconds(path: Path) -> float:
    """Read duration without loading the audio into memory."""
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / wav.getframerate()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(probe.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Batch LatentSync inference")
    parser.add_argument("--input_dir", required=True, help="Folder containing matching audio/video files")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--unet_config_path", default="configs/unet/stage2_512.yaml")
    parser.add_argument("--inference_ckpt_path", default="checkpoints/latentsync_unet.pt")
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--temp_dir", default="temp")
    parser.add_argument("--enable_deepcache", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio = {p.stem: p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS}
    video = {p.stem: p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS}
    pairs = sorted(set(audio) & set(video))
    if not pairs:
        raise SystemExit(f"No matching audio/video pairs found in {input_dir}")

    rows = []
    successful_runtime = 0.0
    successful_audio_duration = 0.0
    for stem in pairs:
        metrics_path = output_dir / f".{stem}.metrics.json"
        output_path = output_dir / f"{stem}_lipsync.mp4"
        command = [sys.executable, "-m", "scripts.inference",
                   "--unet_config_path", args.unet_config_path,
                   "--inference_ckpt_path", args.inference_ckpt_path,
                   "--inference_steps", str(args.inference_steps),
                   "--guidance_scale", str(args.guidance_scale),
                   "--seed", str(args.seed), "--temp_dir", args.temp_dir,
                   "--video_path", str(video[stem]), "--audio_path", str(audio[stem]),
                   "--video_out_path", str(output_path), "--metrics_json", str(metrics_path)]
        if args.enable_deepcache:
            command.append("--enable_deepcache")
        print(f"[{len(rows) + 1}/{len(pairs)}] {stem}")
        result = subprocess.run(command, text=True)
        if result.returncode != 0:
            rows.append({"case": stem, "status": "failed", "error_code": result.returncode})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_path.unlink(missing_ok=True)
        duration = audio_duration_seconds(audio[stem])
        successful_runtime += float(metrics["elapsed_seconds"])
        successful_audio_duration += duration
        rows.append({"case": stem, "status": "success", "audio_duration_seconds": duration, **metrics})

    report = output_dir / "batch_report.csv"
    fields = [
        "case",
        "status",
        "audio_duration_seconds",
        "elapsed_seconds",
        "peak_vram_mb",
        "video_out_path",
        "error_code",
    ]
    with report.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    successful_rows = [r for r in rows if r["status"] == "success"]
    peak = max((float(r["peak_vram_mb"]) for r in successful_rows), default=0)
    ratio = successful_runtime / successful_audio_duration if successful_audio_duration else 0
    print(f"成功样本数: {len(successful_rows)}/{len(rows)}")
    print(f"成功样本总运行时长: {successful_runtime:.2f} 秒")
    print(f"成功样本音频总时长: {successful_audio_duration:.2f} 秒")
    print(f"总运行时长 / 对齐音频时长: {ratio:.4f}")
    print(f"峰值 VRAM: {peak:.2f} MB")
    print(f"报告: {report}")


if __name__ == "__main__":
    main()
