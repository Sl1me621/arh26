import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


DEFAULT_STREAM_URL = "rtsp://10.42.0.1:8554/camera"
WINDOW_NAME = "Drone camera"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "recordings"


def open_stream(url: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(url)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def create_writer(output_path: Path, fps: float, frame: object) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), codec, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video file: {output_path}")

    print(f"Recording started: {output_path}")
    print(f"Video size: {width}x{height}, fps: {fps}")
    return writer


def show_stream(
    url: str,
    reconnect_delay: float,
    output_path: Path,
    fps: float,
) -> None:
    print(f"Connecting to stream: {url}")
    print("Press Q or Esc to close the window.")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = open_stream(url)
    writer = None
    last_error_time = 0.0

    try:
        while True:
            if not capture.isOpened():
                now = time.monotonic()
                if now - last_error_time >= reconnect_delay:
                    print("Stream is not available. Reconnecting...")
                    last_error_time = now

                capture.release()
                time.sleep(reconnect_delay)
                capture = open_stream(url)
                continue

            ok, frame = capture.read()
            if not ok or frame is None:
                now = time.monotonic()
                if now - last_error_time >= reconnect_delay:
                    print("Frame was not received. Reconnecting...")
                    last_error_time = now

                capture.release()
                time.sleep(reconnect_delay)
                capture = open_stream(url)
                continue

            if writer is None:
                writer = create_writer(output_path=output_path, fps=fps, frame=frame)

            writer.write(frame)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if writer is not None:
            writer.release()
            print(f"Recording saved: {output_path}")

        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a Wi-Fi camera stream and show it on this laptop."
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_STREAM_URL,
        help=f"Camera stream URL. Default: {DEFAULT_STREAM_URL}",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="Seconds between reconnect attempts. Default: 1.0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output MP4 path. By default, a timestamped file is saved to recordings.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="FPS for saved video. Default: 30.0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output
    if output_path is None:
        output_path = DEFAULT_OUTPUT_DIR / f"drone_{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"

    show_stream(
        url=args.url,
        reconnect_delay=args.reconnect_delay,
        output_path=output_path.resolve(),
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
