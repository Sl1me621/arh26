"""Run YOLO inference on a video, display detections, and save the result."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT_DIR / "model" / "best.pt"
DEFAULT_SOURCE = PROJECT_DIR / "flight.mp4"
DEFAULT_OUTPUT = PROJECT_DIR / "flight_detected.mp4"
WINDOW_NAME = "YOLO detections - Q/Esc to stop"


def probability(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO on a video, show bounding boxes, and save an MP4."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"Path to YOLO weights (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Path to the input video (default: {DEFAULT_SOURCE}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to the output MP4 (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--conf",
        type=probability,
        default=0.25,
        help="Minimum detection confidence, from 0 to 1 (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=probability,
        default=0.70,
        help="IoU threshold for non-maximum suppression (default: 0.70).",
    )
    parser.add_argument(
        "--imgsz",
        type=positive_int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example 'cpu' or '0' for the first GPU.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a preview window; only save the output video.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output video.",
    )
    return parser.parse_args()


def validate_paths(model_path: Path, source_path: Path, output_path: Path, overwrite: bool) -> None:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights were not found: {model_path}")
    if not source_path.is_file():
        raise FileNotFoundError(f"Input video was not found: {source_path}")
    if source_path == output_path:
        raise ValueError("Input and output video paths must be different.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output video already exists: {output_path}\n"
            "Choose another --output path or add --overwrite."
        )


def open_video(source_path: Path) -> tuple[cv2.VideoCapture, float, int]:
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open input video: {source_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return capture, fps, max(frame_count, 0)


def create_writer(output_path: Path, fps: float, frame: object) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    return writer


def configure_preview(width: int, height: int) -> None:
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    scale = min(1.0, 1280 / width, 720 / height)
    cv2.resizeWindow(WINDOW_NAME, round(width * scale), round(height * scale))


def load_model(model_path: Path) -> object:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise RuntimeError(
            "The 'ultralytics' package is not installed.\n"
            "Install it with: python -m pip install ultralytics"
        ) from None
    return YOLO(str(model_path))


def run(args: argparse.Namespace) -> None:
    model_path = args.model.expanduser().resolve()
    source_path = args.source.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    validate_paths(model_path, source_path, output_path, args.overwrite)

    print(f"Loading model: {model_path}")
    model = load_model(model_path)
    capture, fps, frame_count = open_video(source_path)
    writer = None
    processed_frames = 0
    started_at = time.monotonic()

    print(f"Input:  {source_path}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps:.2f}; frames: {frame_count or 'unknown'}")
    if not args.no_show:
        print("Press Q or Esc in the preview window to stop early.")

    predict_options = {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "verbose": False,
    }
    if args.device is not None:
        predict_options["device"] = args.device

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            result = model.predict(source=frame, **predict_options)[0]
            annotated_frame = result.plot()

            if writer is None:
                writer = create_writer(output_path, fps, annotated_frame)
                if not args.no_show:
                    height, width = annotated_frame.shape[:2]
                    configure_preview(width, height)

            writer.write(annotated_frame)
            processed_frames += 1

            if not args.no_show:
                cv2.imshow(WINDOW_NAME, annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("Processing stopped by the user.")
                    break

            if processed_frames % 100 == 0:
                if frame_count:
                    print(f"Processed {processed_frames}/{frame_count} frames")
                else:
                    print(f"Processed {processed_frames} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    if processed_frames == 0:
        raise RuntimeError("The input video contains no readable frames.")

    elapsed = time.monotonic() - started_at
    print(
        f"Done: {processed_frames} frames in {elapsed:.1f} s "
        f"({processed_frames / elapsed:.1f} frames/s)."
    )
    print(f"Saved to: {output_path}")


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
