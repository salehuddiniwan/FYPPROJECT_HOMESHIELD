"""
Fire detection using local YOLO weights.

Runs inference on images, videos, or your webcam.

Usage examples
--------------
# Image (single file or folder of images)
python detect_fire.py --weights fire_detector.pt --source path/to/image.jpg
python detect_fire.py --weights fire_detector.pt --source path/to/folder

# Video
python detect_fire.py --weights fire_detector.pt --source path/to/video.mp4

# Webcam (device 0 by default)
python detect_fire.py --weights fire_detector.pt --source 0 --show

# Adjust confidence threshold and pick output folder
python detect_fire.py --weights fire_detector.pt --source image.jpg --conf 0.4 --out runs/fire
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_model(weights_path: str):
    """Load the YOLO model from a local weights file."""
    try:
        from ultralytics import YOLO
    except ImportError as e:
        sys.exit(
            "Missing dependency 'ultralytics'. Install it with:\n"
            "    pip install ultralytics\n"
            f"(original error: {e})"
        )
    return YOLO(weights_path)


def parse_source(src: str):
    """Treat purely numeric source as a webcam device index."""
    if src.isdigit():
        return int(src)
    return src


def main():
    parser = argparse.ArgumentParser(description="YOLO fire detection inference")
    parser.add_argument(
        "--weights",
        type=str,
        default="best.pt",
        help="Path to the local YOLO weights file (e.g., fire_detector.pt)",
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, folder, video path, or webcam index (e.g. 0)",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument(
        "--out",
        default="runs/fire",
        help="Folder to save annotated outputs",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display results in a window (good for webcam)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device, e.g. 'cpu', '0', '0,1'. Default: auto",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated outputs to disk",
    )
    args = parser.parse_args()

    # Verify the local weights file exists
    if not Path(args.weights).is_file():
        sys.exit(f"[error] Weights file '{args.weights}' not found. Please provide a valid path using --weights.")

    print(f"[info] Loading local model weights from: {args.weights}")
    model = load_model(args.weights)

    source = parse_source(args.source)
    out_dir = Path(args.out)
    out_dir.parent.mkdir(parents=True, exist_ok=True)

    print(f"[info] Running inference on: {source}")
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=not args.no_save,
        show=args.show,
        project=str(out_dir.parent) if out_dir.parent != Path("") else "runs",
        name=out_dir.name,
        exist_ok=True,
        stream=False,
    )

    # Quick textual summary
    total_dets = 0
    for r in results:
        if r.boxes is not None:
            total_dets += len(r.boxes)
    print(f"[done] {len(results)} frame(s) processed, {total_dets} detection(s) total.")
    if not args.no_save:
        print(f"[done] Annotated output saved under: {out_dir.resolve()}")


if __name__ == "__main__":
    main()