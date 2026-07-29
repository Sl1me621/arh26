import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIRS = (PROJECT_ROOT / "recordings", PROJECT_ROOT / "recording")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dataset"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mkv", ".webm"}


def find_default_input_dir() -> Path:
    for input_dir in DEFAULT_INPUT_DIRS:
        if input_dir.is_dir():
            return input_dir

    expected = " or ".join(str(path) for path in DEFAULT_INPUT_DIRS)
    raise FileNotFoundError(f"Recording directory was not found: {expected}")


def find_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Recording directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Recording path is not a directory: {input_dir}")

    videos = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise FileNotFoundError(
            f"No video files were found in {input_dir}. Supported: {extensions}"
        )

    return videos


def extract_frames(
    video_path: Path,
    output_dir: Path,
    every_nth_frame: int,
    image_format: str,
    filename_prefix: str,
) -> int:
    if every_nth_frame < 1:
        raise ValueError("--every-nth-frame must be greater than or equal to 1")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    saved_count = 0
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % every_nth_frame == 0:
                saved_count += 1
                frame_name = f"{filename_prefix}_frame_{saved_count:06d}.{image_format}"
                frame_path = output_dir / frame_name

                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not write frame: {frame_path}")

            frame_index += 1
    finally:
        capture.release()

    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames into the dataset directory."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Path to one video file. By default, all videos from the recording directory are used.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with recordings. Default: recordings, or recording if recordings does not exist.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for extracted frames. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=1,
        help="Save every Nth frame. Default: 1, saves all frames.",
    )
    parser.add_argument(
        "--format",
        choices=("jpg", "png"),
        default="jpg",
        help="Output image format. Default: jpg.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.video and args.input_dir:
        raise ValueError("Use either --video or --input-dir, not both")

    video_paths = (
        [args.video.resolve()]
        if args.video
        else find_videos((args.input_dir or find_default_input_dir()).resolve())
    )
    output_dir = args.output_dir.resolve()

    total_saved_count = 0

    print(f"Output directory: {output_dir}")
    for video_path in video_paths:
        saved_count = extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            every_nth_frame=args.every_nth_frame,
            image_format=args.format,
            filename_prefix=video_path.stem,
        )
        total_saved_count += saved_count
        print(f"{video_path}: saved {saved_count} frames")

    print(f"Total saved frames: {total_saved_count}")


if __name__ == "__main__":
    main()
