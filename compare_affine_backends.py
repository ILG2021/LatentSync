"""Compare current face alignment against the official LatentSync 106-point path.

This is a temporary regression tool. InsightFace is imported lazily and is not a
project dependency because its pretrained model weights are not licensed for
commercial use.

Example:

    python compare_affine_backends.py --video assets/demo1_video.mp4 --device cuda

After inspecting the first report, the measured values can be turned into CI
limits, for example:

    python compare_affine_backends.py --video input.mp4 \
        --max-p95-anchor-nrmse 0.05 --max-p95-crop-mae 25
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path

import cv2
import numpy as np

from latentsync.utils.affine_transform import AlignRestore
from latentsync.utils.face_detector import FaceDetector, cuda_to_int
from latentsync.utils.image_processor import (
    FACE_ANCHOR_SCALE,
    LEFT_BROW_LANDMARKS,
    NOSE_LANDMARKS,
    RIGHT_BROW_LANDMARKS,
    alignment_anchors,
    interpolate_missing_landmarks,
)


LEGACY_LEFT_BROW = [43, 48, 49, 51, 50]
LEGACY_RIGHT_BROW = [101, 102, 103, 104, 105]
LEGACY_NOSE = [74, 77, 83, 86]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare current YuNet/MediaPipe alignment with legacy InsightFace 106 alignment."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--video", help="One input video used by both preprocessing paths")
    inputs.add_argument("--input-dir", help="Directory containing videos to compare")
    parser.add_argument("--recursive", action="store_true", help="Search input directory recursively")
    parser.add_argument(
        "--extensions",
        default=".mp4,.mov,.mkv,.avi,.webm",
        help="Comma-separated extensions used by --input-dir",
    )
    parser.add_argument("--output-dir", default="affine_comparison")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:0, ...")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--detection-interval", type=int, default=5)
    parser.add_argument(
        "--anchor-scale",
        type=float,
        default=None,
        help=f"Contract current anchors about their centroid (default: {FACE_ANCHOR_SCALE})",
    )
    parser.add_argument("--max-frames", type=int, default=300, help="0 means all frames")
    parser.add_argument("--stride", type=int, default=1, help="Read every Nth source frame")
    parser.add_argument("--visualizations", type=int, default=12)
    parser.add_argument("--insightface-root", default="checkpoints/auxiliary")
    parser.add_argument("--max-interpolation-gap", type=int, default=25)
    parser.add_argument("--fail-fast", action="store_true", help="Stop a folder run after its first error")
    parser.add_argument(
        "--max-p95-anchor-nrmse",
        type=float,
        default=None,
        help="Optional failure threshold for p95 anchor error / legacy eyebrow distance",
    )
    parser.add_argument(
        "--max-p95-crop-mae",
        type=float,
        default=None,
        help="Optional failure threshold for p95 RGB crop mean absolute error (0..255)",
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.detection_interval < 1:
        parser.error("--detection-interval must be at least 1")
    if args.anchor_scale is None:
        args.anchor_scale = FACE_ANCHOR_SCALE
    if not np.isfinite(args.anchor_scale) or args.anchor_scale <= 0:
        parser.error("--anchor-scale must be finite and greater than zero")
    return args


def iter_frames(video_path: Path, stride: int, max_frames: int):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_index = 0
    emitted = 0
    try:
        while max_frames == 0 or emitted < max_frames:
            ok, bgr = capture.read()
            if not ok:
                break
            if source_index % stride == 0:
                yield source_index, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                emitted += 1
            source_index += 1
    finally:
        capture.release()


class LegacyInsightFace106:
    """The detector selection and 106-point output used by the old project code."""

    def __init__(self, root: Path, device: str):
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError(
                "This comparison requires the temporary test dependency 'insightface'. "
                "Install it only in the test environment, then remove it after the comparison."
            ) from error

        if not str(device).startswith("cuda"):
            raise ValueError("The legacy implementation only supported a CUDA device")
        self.app = FaceAnalysis(
            allowed_modules=["detection", "landmark_2d_106"],
            root=str(root),
            providers=["CUDAExecutionProvider"],
        )
        self.app.prepare(ctx_id=cuda_to_int(device), det_size=(512, 512))

    def __call__(self, rgb_frame, threshold=0.5):
        # Deliberately pass RGB exactly as the old LatentSync implementation did.
        faces = self.app.get(rgb_frame)
        selected = None
        selected_area = 0
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            width, height = x2 - x1, y2 - y1
            if width < 50 or height < 80:
                continue
            if width / height > 1.5 or width / height < 0.2:
                continue
            if face.det_score < threshold:
                continue
            if width * height > selected_area:
                selected = face
                selected_area = width * height
        if selected is None:
            return None
        return np.round(selected.landmark_2d_106).astype(np.int32)


def detect_legacy(video_path, stride, max_frames, root, device):
    detector = LegacyInsightFace106(root, device)
    source_indices = []
    landmarks = []
    for source_index, frame in iter_frames(video_path, stride, max_frames):
        source_indices.append(source_index)
        landmarks.append(detector(frame))
    del detector
    gc.collect()
    if not landmarks:
        raise RuntimeError(f"No frames were decoded from: {video_path}")
    return source_indices, landmarks


def detect_current(video_path, stride, max_frames, device, detection_interval):
    detector = FaceDetector(device=device, detection_interval=detection_interval)
    detector.reset_tracking()
    source_indices = []
    landmarks = []
    try:
        for source_index, frame in iter_frames(video_path, stride, max_frames):
            source_indices.append(source_index)
            _, points = detector(frame)
            landmarks.append(points)
    finally:
        detector.close()
        del detector
        gc.collect()
    return source_indices, landmarks


def repair_track(name, landmarks, max_gap):
    missing_before = sum(points is None for points in landmarks)
    failed_index, repaired_count = interpolate_missing_landmarks(landmarks, max_gap=max_gap)
    if failed_index is not None:
        raise RuntimeError(
            f"{name} has more than {max_gap} consecutive missing frames, starting at sampled "
            f"frame {failed_index}"
        )
    return missing_before, repaired_count


def legacy_anchors(points):
    return np.asarray(
        [
            points[LEGACY_LEFT_BROW].mean(axis=0),
            points[LEGACY_RIGHT_BROW].mean(axis=0),
            points[LEGACY_NOSE].mean(axis=0),
        ],
        dtype=np.float32,
    ).round()


def current_anchors(points, scale):
    if scale == FACE_ANCHOR_SCALE:
        # Exercise the exact production helper for the default comparison.
        return alignment_anchors(points)
    points = np.asarray(points, dtype=np.float32)
    anchors = np.asarray(
        [
            points[LEFT_BROW_LANDMARKS].mean(axis=0),
            points[RIGHT_BROW_LANDMARKS].mean(axis=0),
            points[NOSE_LANDMARKS].mean(axis=0),
        ],
        dtype=np.float32,
    )
    centre = anchors.mean(axis=0, keepdims=True)
    return np.round(centre + scale * (anchors - centre))


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def source_crop_corners(matrix, face_size):
    width, height = face_size
    crop_corners = np.asarray(
        [[[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]], dtype=np.float32
    )
    return cv2.transform(crop_corners, cv2.invertAffineTransform(matrix))[0]


def affine_scale(matrix):
    """Return the isotropic source-to-crop scale encoded by a 2x3 matrix."""
    return float(np.hypot(matrix[0, 0], matrix[1, 0]))


def draw_anchor_sets(panel, old_anchors, new_anchors, matrix, face_size):
    """Draw both source anchor triangles in one aligned-crop coordinate system."""
    output_h, output_w = panel.shape[:2]
    scale_to_panel = np.asarray(
        [output_w / face_size[0], output_h / face_size[1]], dtype=np.float32
    )
    names = ("L", "R", "N")
    sets = (
        (old_anchors, (0, 255, 0)),      # RGB green: legacy InsightFace
        (new_anchors, (255, 0, 255)),    # RGB magenta: current MediaPipe
    )
    for anchors, color in sets:
        mapped = cv2.transform(np.asarray([anchors], dtype=np.float32), matrix)[0]
        points = np.rint(mapped * scale_to_panel).astype(np.int32)
        cv2.polylines(panel, [points], True, color, 3, cv2.LINE_AA)
        for name, point in zip(names, points):
            x = int(np.clip(point[0], 0, output_w - 1))
            y = int(np.clip(point[1], 0, output_h - 1))
            cv2.circle(panel, (x, y), 7, color, -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                name,
                (min(x + 9, output_w - 18), max(y - 9, 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )


def compare_tracks(video_path, source_indices, legacy_landmarks, current_landmarks, args):
    legacy_aligner = AlignRestore(resolution=args.resolution, device=args.device)
    current_aligner = AlignRestore(resolution=args.resolution, device=args.device)
    legacy_aligner.p_bias = None
    current_aligner.p_bias = None
    if legacy_aligner.face_size != current_aligner.face_size:
        raise RuntimeError(
            f"Affine face sizes differ: legacy={legacy_aligner.face_size}, "
            f"current={current_aligner.face_size}"
        )

    rows = []
    visualizations = []
    stream = iter_frames(video_path, args.stride, args.max_frames)
    for position, (source_index, frame) in enumerate(stream):
        if position >= len(source_indices) or source_index != source_indices[position]:
            raise RuntimeError("Video frame sequence changed between comparison passes")

        old_anchors = legacy_anchors(legacy_landmarks[position])
        new_anchors = current_anchors(current_landmarks[position], args.anchor_scale)
        old_crop, old_matrix = legacy_aligner.align_warp_face(
            frame.copy(), landmarks3=old_anchors, smooth=True
        )
        new_crop, new_matrix = current_aligner.align_warp_face(
            frame.copy(), landmarks3=new_anchors, smooth=True
        )
        old_crop = cv2.resize(
            old_crop, (args.resolution, args.resolution), interpolation=cv2.INTER_LANCZOS4
        )
        new_crop = cv2.resize(
            new_crop, (args.resolution, args.resolution), interpolation=cv2.INTER_LANCZOS4
        )
        old_matrix = old_matrix.squeeze(0).detach().float().cpu().numpy()
        new_matrix = new_matrix.squeeze(0).detach().float().cpu().numpy()

        baseline_scale = max(float(np.linalg.norm(old_anchors[1] - old_anchors[0])), 1.0)
        anchor_distances = np.linalg.norm(new_anchors - old_anchors, axis=1)
        anchor_rmse = float(np.sqrt(np.mean(np.square(anchor_distances))))
        old_float = old_crop.astype(np.float32)
        new_float = new_crop.astype(np.float32)
        crop_mae = float(np.mean(np.abs(new_float - old_float)))
        crop_mse = float(np.mean(np.square(new_float - old_float)))
        crop_psnr = 100.0 if crop_mse == 0 else 20.0 * math.log10(255.0 / math.sqrt(crop_mse))
        old_corners = source_crop_corners(old_matrix, legacy_aligner.face_size)
        new_corners = source_crop_corners(new_matrix, current_aligner.face_size)
        corner_error = float(np.mean(np.linalg.norm(new_corners - old_corners, axis=1)))
        scale_ratio = affine_scale(new_matrix) / max(affine_scale(old_matrix), 1e-12)

        row = {
            "sample_index": position,
            "source_frame": source_index,
            "left_brow_error_px": float(anchor_distances[0]),
            "right_brow_error_px": float(anchor_distances[1]),
            "nose_error_px": float(anchor_distances[2]),
            "anchor_rmse_px": anchor_rmse,
            "anchor_nrmse": anchor_rmse / baseline_scale,
            "crop_corner_error_px": corner_error,
            "crop_corner_nerror": corner_error / baseline_scale,
            "affine_scale_ratio": scale_ratio,
            "crop_mae": crop_mae,
            "crop_psnr_db": crop_psnr,
        }
        rows.append(row)

        if args.visualizations > 0:
            source = cv2.resize(frame, old_crop.shape[1::-1], interpolation=cv2.INTER_AREA)
            old_crop_overlay = old_crop.copy()
            new_crop_overlay = new_crop.copy()
            draw_anchor_sets(
                old_crop_overlay,
                old_anchors,
                new_anchors,
                old_matrix,
                legacy_aligner.face_size,
            )
            draw_anchor_sets(
                new_crop_overlay,
                old_anchors,
                new_anchors,
                new_matrix,
                current_aligner.face_size,
            )
            difference = cv2.absdiff(old_crop, new_crop)
            difference = cv2.applyColorMap(
                cv2.cvtColor(difference, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_TURBO
            )
            difference = cv2.cvtColor(difference, cv2.COLOR_BGR2RGB)
            panels = [source, old_crop_overlay, new_crop_overlay, difference]
            labels = [
                f"source | anchor={args.anchor_scale:.3f} affine(new/old)={scale_ratio:.4f}",
                "legacy | green=old magenta=new",
                "current | green=old magenta=new",
                "absolute difference",
            ]
            for panel, label in zip(panels, labels):
                cv2.putText(
                    panel, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.putText(
                    panel, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1
                )
            visualizations.append((crop_mae, source_index, np.concatenate(panels, axis=1)))
            visualizations = sorted(visualizations, key=lambda item: item[0], reverse=True)[
                : args.visualizations
            ]

    if len(rows) != len(source_indices):
        raise RuntimeError(
            f"Expected {len(source_indices)} comparison frames, decoded {len(rows)} on final pass"
        )
    return rows, visualizations


def summarize(rows, legacy_missing, current_missing, args, video_path):
    metrics = [
        "left_brow_error_px",
        "right_brow_error_px",
        "nose_error_px",
        "anchor_rmse_px",
        "anchor_nrmse",
        "crop_corner_error_px",
        "crop_corner_nerror",
        "affine_scale_ratio",
        "crop_mae",
        "crop_psnr_db",
    ]
    summary = {
        "video": str(video_path.resolve()),
        "sampled_frames": len(rows),
        "stride": args.stride,
        "resolution": args.resolution,
        "current_detection_interval": args.detection_interval,
        "current_anchor_scale": args.anchor_scale,
        "comparison_scope": (
            "Legacy InsightFace and current landmarks rendered through the same current "
            "AlignRestore implementation, isolating landmark/crop-geometry differences"
        ),
        "legacy_missing_before_interpolation": legacy_missing,
        "current_missing_before_interpolation": current_missing,
        "metrics": {},
    }
    for metric in metrics:
        values = [row[metric] for row in rows if math.isfinite(row[metric])]
        summary["metrics"][metric] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "p05": percentile(values, 5),
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
            "max": float(np.max(values)),
        }
    return summary


def write_visualizations(output_dir, visualizations):
    visual_dir = output_dir / "worst_frames"
    visual_dir.mkdir(parents=True, exist_ok=True)
    for rank, (crop_mae, source_frame, contact_sheet) in enumerate(visualizations, start=1):
        filename = visual_dir / f"{rank:02d}_frame_{source_frame:06d}_mae_{crop_mae:.2f}.jpg"
        bgr = cv2.cvtColor(contact_sheet, cv2.COLOR_RGB2BGR)
        encoded_ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not encoded_ok:
            raise RuntimeError(f"OpenCV could not encode visualization: {filename}")
        # cv2.imwrite silently fails for some non-ASCII paths on Windows. Let
        # pathlib perform the filesystem write so Chinese video names work.
        filename.write_bytes(encoded.tobytes())
    return len(visualizations)


def write_report(output_dir, rows, summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "per_frame.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def threshold_failures(summary, args):
    failures = []
    anchor_p95 = summary["metrics"]["anchor_nrmse"]["p95"]
    crop_p95 = summary["metrics"]["crop_mae"]["p95"]
    if args.max_p95_anchor_nrmse is not None and anchor_p95 > args.max_p95_anchor_nrmse:
        failures.append(f"anchor_nrmse p95 {anchor_p95:.6f} > {args.max_p95_anchor_nrmse:.6f}")
    if args.max_p95_crop_mae is not None and crop_p95 > args.max_p95_crop_mae:
        failures.append(f"crop_mae p95 {crop_p95:.3f} > {args.max_p95_crop_mae:.3f}")
    return failures


def discover_videos(args):
    if args.video:
        video = Path(args.video)
        if not video.is_file():
            raise FileNotFoundError(f"Input video does not exist: {video}")
        return [(video, Path(video.stem))]

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    extensions = {
        extension.strip().lower() if extension.strip().startswith(".") else f".{extension.strip().lower()}"
        for extension in args.extensions.split(",")
        if extension.strip()
    }
    iterator = input_dir.rglob("*") if args.recursive else input_dir.glob("*")
    videos = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in extensions)
    if not videos:
        raise RuntimeError(f"No matching videos found in: {input_dir}")
    results = []
    for video in videos:
        relative = video.relative_to(input_dir)
        # Preserve the extension in the report directory name so sibling files
        # such as presenter.mp4 and presenter.mov never overwrite each other.
        report_name = f"{relative.stem}_{relative.suffix.removeprefix('.')}"
        results.append((video, relative.parent / report_name))
    return results


def run_comparison(video_path, output_dir, args):
    print(f"\n=== {video_path} ===")
    print("Running legacy InsightFace 106-point preprocessing...")
    source_indices, legacy_landmarks = detect_legacy(
        video_path,
        args.stride,
        args.max_frames,
        Path(args.insightface_root),
        args.device,
    )
    print(f"Loaded {len(source_indices)} sampled frames")
    legacy_missing, legacy_repaired = repair_track(
        "Legacy InsightFace", legacy_landmarks, args.max_interpolation_gap
    )

    print(
        f"Running current preprocessing with YuNet interval {args.detection_interval}, "
        f"anchor scale {args.anchor_scale:g}..."
    )
    current_indices, current_landmarks = detect_current(
        video_path, args.stride, args.max_frames, args.device, args.detection_interval
    )
    if current_indices != source_indices:
        raise RuntimeError("Video frame sequence changed between detector passes")
    current_missing, current_repaired = repair_track(
        "Current YuNet/MediaPipe", current_landmarks, args.max_interpolation_gap
    )

    print("Rendering both affine tracks...")
    rows, visualizations = compare_tracks(
        video_path, source_indices, legacy_landmarks, current_landmarks, args
    )
    summary = summarize(rows, legacy_missing, current_missing, args, video_path)
    summary["legacy_repaired_frames"] = legacy_repaired
    summary["current_repaired_frames"] = current_repaired
    failures = threshold_failures(summary, args)
    summary["threshold_failures"] = failures
    visualizations_written = write_visualizations(output_dir, visualizations)
    summary["visualizations_written"] = visualizations_written
    write_report(output_dir, rows, summary)

    anchor = summary["metrics"]["anchor_nrmse"]
    crop = summary["metrics"]["crop_mae"]
    corners = summary["metrics"]["crop_corner_nerror"]
    scale = summary["metrics"]["affine_scale_ratio"]
    print(f"anchor NRMSE: mean={anchor['mean']:.4f}, p95={anchor['p95']:.4f}, max={anchor['max']:.4f}")
    print(f"crop corner normalized error: mean={corners['mean']:.4f}, p95={corners['p95']:.4f}")
    print(
        f"current/legacy affine scale: mean={scale['mean']:.4f}, "
        f"median={scale['p50']:.4f}, p95={scale['p95']:.4f}"
    )
    print(f"crop RGB MAE: mean={crop['mean']:.2f}, p95={crop['p95']:.2f}, max={crop['max']:.2f}")
    print(f"Worst-frame visualizations written: {visualizations_written}")
    print(f"Report written to {output_dir.resolve()}")
    return summary


def write_batch_report(output_dir, results):
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "videos_total": len(results),
        "videos_passed": sum(result["status"] == "passed" for result in results),
        "videos_failed": sum(result["status"] != "passed" for result in results),
        "results": results,
    }
    with (output_dir / "batch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    rows = []
    for result in results:
        row = {
            "video": result["video"],
            "status": result["status"],
            "error": result.get("error", ""),
            "sampled_frames": result.get("sampled_frames", ""),
            "legacy_missing": result.get("legacy_missing_before_interpolation", ""),
            "current_missing": result.get("current_missing_before_interpolation", ""),
            "current_anchor_scale": result.get("current_anchor_scale", ""),
            "anchor_nrmse_p95": "",
            "crop_corner_nerror_p95": "",
            "affine_scale_ratio_p05": "",
            "affine_scale_ratio_p50": "",
            "affine_scale_ratio_p95": "",
            "crop_mae_p95": "",
        }
        if "metrics" in result:
            row["anchor_nrmse_p95"] = result["metrics"]["anchor_nrmse"]["p95"]
            row["crop_corner_nerror_p95"] = result["metrics"]["crop_corner_nerror"]["p95"]
            row["affine_scale_ratio_p05"] = result["metrics"]["affine_scale_ratio"]["p05"]
            row["affine_scale_ratio_p50"] = result["metrics"]["affine_scale_ratio"]["p50"]
            row["affine_scale_ratio_p95"] = result["metrics"]["affine_scale_ratio"]["p95"]
            row["crop_mae_p95"] = result["metrics"]["crop_mae"]["p95"]
        rows.append(row)
    with (output_dir / "per_video.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    videos = discover_videos(args)
    folder_mode = args.input_dir is not None
    results = []
    for video_path, relative_output in videos:
        video_output = output_dir / relative_output if folder_mode else output_dir
        try:
            summary = run_comparison(video_path, video_output, args)
            summary["status"] = "failed" if summary["threshold_failures"] else "passed"
            results.append(summary)
        except Exception as error:
            result = {"video": str(video_path.resolve()), "status": "error", "error": str(error)}
            results.append(result)
            print(f"ERROR: {video_path}: {error}")
            if args.fail_fast or not folder_mode:
                write_batch_report(output_dir, results)
                raise

    write_batch_report(output_dir, results)
    failed = sum(result["status"] != "passed" for result in results)
    print(f"\nBatch report: {output_dir.resolve()}")
    print(f"Videos: {len(results)}, passed: {len(results) - failed}, failed: {failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
