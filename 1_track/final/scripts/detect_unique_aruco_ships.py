import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARUCO_DIR = PROJECT_ROOT / "aruco"
DEFAULT_INPUT_DIRS = (DEFAULT_ARUCO_DIR,)
DEFAULT_VIDEO_PATH = DEFAULT_ARUCO_DIR / "aruco_flight.mp4"
DEFAULT_OUTPUT_DIR = DEFAULT_ARUCO_DIR / "aruco_detections"
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mov", ".mp4", ".mkv", ".webm"}
DEFAULT_DICTIONARY = "DICT_4X4_1000"
DEFAULT_ALLOWED_MARKER_IDS = (15, 20, 22, 36, 32, 25, 7)
DEFAULT_MIN_DETECTIONS = 1
DEFAULT_DISPLAY_DELAY_MS = 1


@dataclass
class ShipDetection:
    marker_id: int
    first_frame_number: int
    snapshot_path: Path
    detections_in_video: int = 0


def get_aruco_detector(dictionary_name: str) -> object:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install opencv-contrib-python."
        )

    if not hasattr(cv2.aruco, dictionary_name):
        available = ", ".join(
            name for name in dir(cv2.aruco) if name.startswith("DICT_")
        )
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}. Available: {available}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters


def detect_aruco_markers(frame: object, detector: object) -> tuple[list[int], object]:
    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        dictionary, parameters = detector
        corners, ids, _ = cv2.aruco.detectMarkers(
            frame,
            dictionary,
            parameters=parameters,
        )

    if ids is None:
        return [], corners

    marker_ids = [int(marker_id) for marker_id in ids.flatten()]
    return marker_ids, corners


def filter_marker_detections(
    marker_ids: list[int],
    corners: object,
    allowed_marker_ids: set[int] | None,
) -> tuple[list[int], list[object]]:
    if allowed_marker_ids is None:
        return marker_ids, list(corners)

    filtered_ids = []
    filtered_corners = []
    for marker_id, marker_corners in zip(marker_ids, corners):
        if marker_id in allowed_marker_ids:
            filtered_ids.append(marker_id)
            filtered_corners.append(marker_corners)

    return filtered_ids, filtered_corners


def draw_detection_overlay(
    frame: object,
    marker_ids: list[int],
    corners: list[object],
    frame_number: int,
) -> object:
    display_frame = frame.copy()
    if corners:
        cv2.aruco.drawDetectedMarkers(display_frame, corners)

    text = f"frame={frame_number}"
    if marker_ids:
        text += f" ids={sorted(set(marker_ids))}"

    cv2.putText(
        display_frame,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0) if marker_ids else (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display_frame


def print_detection_log(
    video_path: Path,
    frame_number: int,
    marker_ids: list[int],
    marker_counts: dict[int, int],
    newly_confirmed_ids: list[int],
) -> None:
    counts = ", ".join(
        f"id={marker_id}: {marker_counts[marker_id]}"
        for marker_id in sorted(set(marker_ids))
    )
    message = (
        f"{video_path.name}: frame={frame_number}; "
        f"detected_ids={sorted(set(marker_ids))}; counts=({counts})"
    )
    if newly_confirmed_ids:
        message += f"; new_unique_ids={sorted(newly_confirmed_ids)}"

    print(message, flush=True)


def recognize_unique_ships(
    frame: object,
    frame_number: int,
    marker_counts: dict[int, int],
    first_marker_frames: dict[int, tuple[int, object, object]],
    confirmed_marker_ids: set[int],
    detections: list[ShipDetection],
    output_dir: Path,
    video_stem: str,
    detector: object,
    min_detections: int,
    allowed_marker_ids: set[int] | None,
) -> tuple[list[int], list[object], list[int]]:
    marker_ids, corners = detect_aruco_markers(frame, detector)
    marker_ids, corners = filter_marker_detections(
        marker_ids,
        corners,
        allowed_marker_ids,
    )

    if not marker_ids:
        return [], [], []

    newly_confirmed_ids = []

    for marker_id in set(marker_ids):
        if marker_id not in first_marker_frames:
            first_marker_frames[marker_id] = (frame_number, frame.copy(), corners)

        marker_counts[marker_id] = marker_counts.get(marker_id, 0) + 1
        if (
            marker_counts[marker_id] < min_detections
            or marker_id in confirmed_marker_ids
        ):
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        first_frame_number, first_frame, first_corners = first_marker_frames[marker_id]
        annotated_frame = first_frame.copy()
        if first_corners:
            cv2.aruco.drawDetectedMarkers(annotated_frame, first_corners)

        snapshot_path = output_dir / (
            f"{video_stem}_id_{marker_id}_frame_{first_frame_number:06d}.jpg"
        )

        if not cv2.imwrite(str(snapshot_path), annotated_frame):
            raise RuntimeError(f"Could not write snapshot: {snapshot_path}")

        confirmed_marker_ids.add(marker_id)
        newly_confirmed_ids.append(marker_id)
        detections.append(
            ShipDetection(
                marker_id=marker_id,
                first_frame_number=first_frame_number,
                snapshot_path=snapshot_path,
                detections_in_video=marker_counts[marker_id],
            )
        )

    return marker_ids, corners, newly_confirmed_ids


def show_detection_frame(
    window_name: str,
    frame: object,
    marker_ids: list[int],
    corners: list[object],
    frame_number: int,
    display_delay_ms: int,
) -> bool:
    display_frame = draw_detection_overlay(frame, marker_ids, corners, frame_number)
    cv2.imshow(window_name, display_frame)
    key = cv2.waitKey(display_delay_ms) & 0xFF
    return key not in {ord("q"), 27}


def find_default_input_dir() -> Path:
    for input_dir in DEFAULT_INPUT_DIRS:
        if input_dir.is_dir():
            return input_dir

    expected = " or ".join(str(path) for path in DEFAULT_INPUT_DIRS)
    raise FileNotFoundError(f"Recording directory was not found: {expected}")


def find_videos(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Video path is not a directory: {input_dir}")

    videos = sorted(
        path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not videos:
        extensions = ", ".join(sorted(VIDEO_EXTENSIONS))
        raise FileNotFoundError(
            f"No video files were found in {input_dir}. Supported: {extensions}"
        )

    return videos


def process_video(
    video_path: Path,
    output_dir: Path,
    detector: object,
    every_nth_frame: int,
    min_detections: int,
    allowed_marker_ids: set[int] | None,
    show_video: bool,
    display_delay_ms: int,
    log_detections: bool,
) -> tuple[list[ShipDetection], dict[int, int]]:
    if every_nth_frame < 1:
        raise ValueError("--every-nth-frame must be greater than or equal to 1")

    if min_detections < 1:
        raise ValueError("--min-detections must be greater than or equal to 1")

    if display_delay_ms < 1:
        raise ValueError("--display-delay-ms must be greater than or equal to 1")

    if not video_path.exists():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    marker_counts: dict[int, int] = {}
    first_marker_frames: dict[int, tuple[int, object, object]] = {}
    confirmed_marker_ids: set[int] = set()
    detections: list[ShipDetection] = []
    frame_index = 0
    window_name = f"ArUco detection: {video_path.name}"

    if show_video:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_number = frame_index + 1
            marker_ids = []
            corners = []
            if frame_index % every_nth_frame == 0:
                marker_ids, corners, newly_confirmed_ids = recognize_unique_ships(
                    frame=frame,
                    frame_number=frame_number,
                    marker_counts=marker_counts,
                    first_marker_frames=first_marker_frames,
                    confirmed_marker_ids=confirmed_marker_ids,
                    detections=detections,
                    output_dir=output_dir,
                    video_stem=video_path.stem,
                    detector=detector,
                    min_detections=min_detections,
                    allowed_marker_ids=allowed_marker_ids,
                )
                if log_detections and marker_ids:
                    print_detection_log(
                        video_path=video_path,
                        frame_number=frame_number,
                        marker_ids=marker_ids,
                        marker_counts=marker_counts,
                        newly_confirmed_ids=newly_confirmed_ids,
                    )

            if show_video and not show_detection_frame(
                window_name=window_name,
                frame=frame,
                marker_ids=marker_ids,
                corners=corners,
                frame_number=frame_number,
                display_delay_ms=display_delay_ms,
            ):
                print("Video preview stopped by user.", flush=True)
                break

            frame_index += 1
    finally:
        capture.release()
        if show_video:
            cv2.destroyWindow(window_name)

    for detection in detections:
        detection.detections_in_video = marker_counts[detection.marker_id]

    return detections, marker_counts


def parse_marker_ids(value: str) -> set[int] | None:
    if value.strip().lower() in {"", "none", "all"}:
        return None

    marker_ids = set()
    for item in value.split(","):
        item = item.strip()
        if item:
            marker_ids.add(int(item))

    return marker_ids or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect unique ships by ArUco marker IDs in MP4 videos."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=f"Path to one video file. Default: {DEFAULT_VIDEO_PATH}",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with videos. Use only when you intentionally want several videos.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for annotated detection frames. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY,
        help=f"OpenCV ArUco dictionary name. Default: {DEFAULT_DICTIONARY}",
    )
    parser.add_argument(
        "--allowed-ids",
        default=",".join(str(marker_id) for marker_id in DEFAULT_ALLOWED_MARKER_IDS),
        help=(
            "Comma-separated marker IDs that may be counted as ships. "
            "Use 'all' to disable this filter. "
            f"Default: {','.join(str(marker_id) for marker_id in DEFAULT_ALLOWED_MARKER_IDS)}"
        ),
    )
    parser.add_argument(
        "--every-nth-frame",
        type=int,
        default=1,
        help="Analyze every Nth frame. Default: 1, analyzes all frames.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=DEFAULT_MIN_DETECTIONS,
        help=(
            "How many frames must contain the same marker before it is counted as a "
            f"unique ship. Default: {DEFAULT_MIN_DETECTIONS}."
        ),
    )
    parser.add_argument(
        "--show-video",
        action="store_true",
        help="Show annotated video while detecting markers. Press q or Esc to stop.",
    )
    parser.add_argument(
        "--display-delay-ms",
        type=int,
        default=DEFAULT_DISPLAY_DELAY_MS,
        help=(
            "Delay for the video preview window in milliseconds. "
            f"Default: {DEFAULT_DISPLAY_DELAY_MS}."
        ),
    )
    parser.add_argument(
        "--no-log-detections",
        action="store_true",
        help="Disable frame-by-frame logs when an allowed marker is detected.",
    )
    return parser.parse_args()


def print_video_report(
    video_path: Path,
    detections: list[ShipDetection],
    marker_counts: dict[int, int],
    allowed_marker_ids: set[int] | None,
) -> None:
    print(f"Video: {video_path}")
    print(f"Unique objects: {len(detections)}")

    if not detections:
        print("No allowed ArUco markers were detected.")
    else:
        for detection in sorted(detections, key=lambda item: item.marker_id):
            print(
                "  "
                f"id={detection.marker_id}; "
                f"first_frame={detection.first_frame_number}; "
                f"detections={detection.detections_in_video}; "
                f"photo={detection.snapshot_path}"
            )

    if allowed_marker_ids is None:
        return

    seen_ids = set(marker_counts)
    missing_ids = sorted(allowed_marker_ids - seen_ids)
    confirmed_ids = {detection.marker_id for detection in detections}
    below_threshold_ids = sorted(seen_ids - confirmed_ids)

    if missing_ids:
        print(f"Allowed IDs not seen in this video: {missing_ids}")

    if below_threshold_ids:
        below_threshold = ", ".join(
            f"id={marker_id}: {marker_counts[marker_id]}"
            for marker_id in below_threshold_ids
        )
        print(f"Seen but below --min-detections: {below_threshold}")


def main() -> None:
    args = parse_args()
    if args.video and args.input_dir:
        raise ValueError("Use either --video or --input-dir, not both")

    detector = get_aruco_detector(args.dictionary)
    allowed_marker_ids = parse_marker_ids(args.allowed_ids)

    if args.video:
        video_paths = [args.video.resolve()]
    elif args.input_dir:
        video_paths = find_videos(args.input_dir.resolve())
    else:
        video_paths = [DEFAULT_VIDEO_PATH.resolve()]

    output_dir = args.output_dir.resolve()

    total_unique_ids: set[int] = set()
    total_seen_ids: set[int] = set()

    print(f"ArUco dictionary: {args.dictionary}")
    print(f"Allowed marker IDs: {sorted(allowed_marker_ids) if allowed_marker_ids else 'all'}")
    print(f"Output directory: {output_dir}")
    print(f"Detection logs: {'off' if args.no_log_detections else 'on'}")
    print(f"Video preview: {'on' if args.show_video else 'off'}")

    for video_path in video_paths:
        video_output_dir = output_dir / video_path.stem
        detections, marker_counts = process_video(
            video_path=video_path,
            output_dir=video_output_dir,
            detector=detector,
            every_nth_frame=args.every_nth_frame,
            min_detections=args.min_detections,
            allowed_marker_ids=allowed_marker_ids,
            show_video=args.show_video,
            display_delay_ms=args.display_delay_ms,
            log_detections=not args.no_log_detections,
        )
        total_unique_ids.update(detection.marker_id for detection in detections)
        total_seen_ids.update(marker_counts)
        print_video_report(video_path, detections, marker_counts, allowed_marker_ids)

    if len(video_paths) > 1:
        print(f"Total unique objects in all videos: {len(total_unique_ids)}")
        print(f"Total unique IDs: {sorted(total_unique_ids)}")
        if allowed_marker_ids is not None:
            print(f"Allowed IDs not seen in all videos: {sorted(allowed_marker_ids - total_seen_ids)}")


if __name__ == "__main__":
    main()
