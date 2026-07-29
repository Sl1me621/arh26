from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_FPS = 30.0
DICTIONARY_NAME = "DICT_4X4_50"
TARGET_ARUCO_ID = 5
WINDOW = "white rectangle viewer"
MASK_WINDOW = "white mask"
TRACKBAR = "Frame"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

WHITE_SATURATION_MAX = 36
WHITE_VALUE_MIN = 132

MIN_WHITE_RECT_AREA_RATIO = 0.03
MAX_WHITE_RECT_AREA_RATIO = 0.08
MAX_WHITE_RECT_ASPECT_RATIO = 1.31
MIN_RECTANGULARITY = 0.2
MIN_RECT_SIDE_PX = 1

CLOSE_KERNEL = 3
CLOSE_ITERATIONS = 2

OPEN_KERNEL = 7
OPEN_ITERATIONS = 1

MARKER_SEARCH_PADDING_RATIO = 0.25
MIN_MARKER_ROI_SIDE_PX = 20


@dataclass(frozen=True)
class RectCandidate:
    box: np.ndarray
    center: tuple[int, int]
    area_px: float
    area_ratio: float
    aspect_ratio: float
    rectangularity: float
    score: float


@dataclass(frozen=True)
class ArucoDetection:
    corners: np.ndarray
    center: tuple[int, int]
    marker_id: int
    roi: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OpenCV white rectangle finder with ArUco ID 5 detection."
    )
    parser.add_argument("video", nargs="?", help="Path to a video file")
    parser.add_argument("--dict", default=DICTIONARY_NAME, help="ArUco dictionary, for example DICT_4X4_50")
    return parser.parse_args()


def find_video_files() -> list[Path]:
    cwd = Path.cwd()
    roots = [cwd]
    if cwd.parent != cwd:
        roots.append(cwd.parent)
        roots.extend(path for path in cwd.parent.iterdir() if path.is_dir() and path != cwd)

    videos: dict[Path, Path] = {}
    for root in roots:
        try:
            for item in root.iterdir():
                if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                    videos[item.resolve()] = item
        except OSError:
            continue

    return sorted(videos.values(), key=lambda path: str(path).lower())


def choose_video(video_arg: str | None) -> Path | None:
    if video_arg:
        return Path(video_arg)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename(
            title="Choose video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if file_name:
            return Path(file_name)
    except Exception as exc:
        print(f"Could not open file dialog: {exc}", file=sys.stderr)

    videos = find_video_files()
    if not videos:
        print("No video files found near the current directory.")
        return None

    if len(videos) == 1:
        print(f"Found video: {videos[0]}")
        return videos[0]

    print("Found several videos:")
    for index, path in enumerate(videos, start=1):
        print(f"  {index}: {path}")

    while True:
        choice = input("Enter video number, or empty to cancel: ").strip()
        if not choice:
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(videos):
                return videos[index - 1]
        print("Wrong number, try again.")


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = DEFAULT_FPS

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, max(0, frame_count)


def check_aruco_module() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV was installed without cv2.aruco. "
            "Install opencv-contrib-python and remove conflicting OpenCV packages."
        )


def create_detector(dictionary_name: str) -> Any:
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"No ArUco dictionary in cv2.aruco: {dictionary_name}")

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    else:
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    for name, value in {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 43,
        "adaptiveThreshWinSizeStep": 8,
        "adaptiveThreshConstant": 9,
        "minMarkerPerimeterRate": 0.025,
        "maxMarkerPerimeterRate": 4,
        "polygonalApproxAccuracyRate": 0.08,
        "minCornerDistanceRate": 0.02,
        "minDistanceToBorder": 2,
        "errorCorrectionRate": 0.8,
    }.items():
        if hasattr(parameters, name):
            setattr(parameters, name, value)

    if hasattr(parameters, "cornerRefinementMethod") and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters


def detect_markers(image: np.ndarray, detector: Any) -> tuple[list[np.ndarray], np.ndarray | None]:
    if hasattr(detector, "detectMarkers"):
        corners, ids, _ = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, _ = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    return list(corners) if corners is not None else [], ids


def marker_area(corners: np.ndarray) -> float:
    points = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return float(abs(cv2.contourArea(points)))


def preprocess_marker_roi(roi: np.ndarray) -> list[np.ndarray]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(gray, 5, 45, 45)
    adaptive = cv2.adaptiveThreshold(
        clahe_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    return [
        gray,
        clahe_gray,
        clahe.apply(bilateral),
        adaptive,
    ]


def candidate_roi(frame: np.ndarray, candidate: RectCandidate) -> tuple[int, int, int, int] | None:
    frame_height, frame_width = frame.shape[:2]
    x, y, width, height = cv2.boundingRect(candidate.box.astype(np.int32))
    padding = int(round(max(width, height) * MARKER_SEARCH_PADDING_RATIO))

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame_width, x + width + padding)
    y2 = min(frame_height, y + height + padding)

    roi_width = x2 - x1
    roi_height = y2 - y1
    if roi_width < MIN_MARKER_ROI_SIDE_PX or roi_height < MIN_MARKER_ROI_SIDE_PX:
        return None

    return x1, y1, roi_width, roi_height


def find_target_aruco_in_candidate(
    frame: np.ndarray,
    candidate: RectCandidate,
    detector: Any,
) -> ArucoDetection | None:
    roi_rect = candidate_roi(frame, candidate)
    if roi_rect is None:
        return None

    x, y, width, height = roi_rect
    roi = frame[y : y + height, x : x + width]

    target_markers: list[np.ndarray] = []
    for image_for_detection in preprocess_marker_roi(roi):
        corners, ids = detect_markers(image_for_detection, detector)
        if ids is None:
            continue
        for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
            if int(marker_id) != TARGET_ARUCO_ID:
                continue
            points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            points += np.array([x, y], dtype=np.float32)
            target_markers.append(points)

    if not target_markers:
        return None

    best_corners = max(target_markers, key=marker_area)
    center = tuple(np.round(best_corners.mean(axis=0)).astype(int))
    return ArucoDetection(
        corners=best_corners,
        center=center,
        marker_id=TARGET_ARUCO_ID,
        roi=roi_rect,
    )


def find_target_aruco(
    frame: np.ndarray,
    candidates: list[RectCandidate],
    detector: Any,
) -> tuple[RectCandidate | None, ArucoDetection | None]:
    for candidate in candidates:
        detection = find_target_aruco_in_candidate(frame, candidate, detector)
        if detection is not None:
            return candidate, detection

    return None, None


def create_white_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, WHITE_VALUE_MIN], dtype=np.uint8)
    upper_white = np.array([179, WHITE_SATURATION_MAX, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    if CLOSE_ITERATIONS > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (CLOSE_KERNEL, CLOSE_KERNEL),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=CLOSE_ITERATIONS,
        )

    if OPEN_ITERATIONS > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (OPEN_KERNEL, OPEN_KERNEL),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=OPEN_ITERATIONS,
        )

    return mask


def find_rectangle_candidates(frame: np.ndarray, mask: np.ndarray) -> list[RectCandidate]:
    frame_height, frame_width = frame.shape[:2]
    frame_area = float(frame_height * frame_width)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[RectCandidate] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue

        area_ratio = contour_area / frame_area
        if area_ratio < MIN_WHITE_RECT_AREA_RATIO:
            continue
        if area_ratio > MAX_WHITE_RECT_AREA_RATIO:
            continue

        rotated_rect = cv2.minAreaRect(contour)
        width, height = rotated_rect[1]
        if width <= 0 or height <= 0:
            continue
        if min(width, height) < MIN_RECT_SIDE_PX:
            continue

        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > MAX_WHITE_RECT_ASPECT_RATIO:
            continue

        rect_area = width * height
        if rect_area <= 0:
            continue

        rectangularity = contour_area / rect_area
        if rectangularity < MIN_RECTANGULARITY:
            continue

        box = cv2.boxPoints(rotated_rect).astype(np.int32)
        center = tuple(np.round(box.mean(axis=0)).astype(int))
        score = area_ratio * rectangularity / max(aspect_ratio, 1.0)
        candidates.append(
            RectCandidate(
                box=box,
                center=center,
                area_px=contour_area,
                area_ratio=area_ratio,
                aspect_ratio=aspect_ratio,
                rectangularity=rectangularity,
                score=score,
            )
        )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def process_frame(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    paused: bool,
    detector: Any,
) -> tuple[np.ndarray, np.ndarray]:
    output = frame.copy()
    mask = create_white_mask(frame)
    candidates = find_rectangle_candidates(frame, mask)
    marker_candidate, marker_detection = find_target_aruco(frame, candidates, detector)
    best_candidate = marker_candidate or (candidates[0] if candidates else None)

    frame_center = (output.shape[1] // 2, output.shape[0] // 2)
    cv2.drawMarker(output, frame_center, (255, 0, 0), cv2.MARKER_CROSS, 25, 2)

    for candidate in candidates:
        cv2.polylines(output, [candidate.box], True, (0, 180, 0), 2)

    if best_candidate is not None:
        best = best_candidate
        cv2.polylines(output, [best.box], True, (0, 255, 255), 5)
        cv2.circle(output, best.center, 7, (0, 0, 255), -1)
        cv2.line(output, frame_center, best.center, (255, 0, 0), 2)

        error_x = best.center[0] - frame_center[0]
        error_y = best.center[1] - frame_center[1]
        title = "RECTANGLE WITH ARUCO ID 5" if marker_detection else "WHITE RECTANGLE FOUND"
        put_text(output, title, (20, 40), (0, 255, 255))
        put_text(output, f"center={best.center} dx={error_x} dy={error_y}", (20, 70), (0, 255, 255))
        put_text(output, f"area={best.area_px:.0f}px ratio={best.area_ratio:.4f}", (20, 100), (0, 255, 255))
        put_text(output, f"aspect={best.aspect_ratio:.2f} rect={best.rectangularity:.2f}", (20, 130), (0, 255, 255))
    else:
        put_text(output, "NO WHITE RECTANGLE", (20, 40), (0, 0, 255))

    if marker_detection is not None:
        marker_points = marker_detection.corners.astype(np.int32).reshape(1, 4, 2)
        cv2.polylines(output, marker_points, True, (0, 255, 0), 4)
        cv2.circle(output, marker_detection.center, 6, (0, 0, 255), -1)
        marker_error_x = marker_detection.center[0] - frame_center[0]
        marker_error_y = marker_detection.center[1] - frame_center[1]
        x, y, width, height = marker_detection.roi
        cv2.rectangle(output, (x, y), (x + width, y + height), (255, 255, 0), 2)
        put_text(output, f"ARUCO ID {TARGET_ARUCO_ID} FOUND", (20, 280), (0, 255, 0))
        put_text(
            output,
            f"aruco_center={marker_detection.center} dx={marker_error_x} dy={marker_error_y}",
            (20, 310),
            (0, 255, 0),
        )
    elif candidates:
        put_text(output, f"ARUCO ID {TARGET_ARUCO_ID} NOT FOUND IN RECT", (20, 280), (0, 0, 255))

    put_text(output, f"candidates={len(candidates)}", (20, 160), (255, 255, 255))
    put_text(output, f"Smax={WHITE_SATURATION_MAX} Vmin={WHITE_VALUE_MIN}", (20, 190), (255, 255, 255))
    put_text(
        output,
        f"area={MIN_WHITE_RECT_AREA_RATIO:.4f}-{MAX_WHITE_RECT_AREA_RATIO:.2f}",
        (20, 220),
        (255, 255, 255),
    )
    put_text(
        output,
        f"aspect<={MAX_WHITE_RECT_ASPECT_RATIO:.2f} rect>={MIN_RECTANGULARITY:.2f}",
        (20, 250),
        (255, 255, 255),
    )

    status = "PAUSED" if paused else "PLAY"
    seconds = frame_index / fps if fps > 0 else 0.0
    put_text(
        output,
        f"{status} frame={frame_index} time={format_time(seconds)}",
        (20, output.shape[0] - 25),
        (255, 255, 255),
    )

    return output, mask


def print_help(video_path: Path, dictionary_name: str) -> None:
    print(f"Video: {video_path}")
    print(f"ArUco: dictionary={dictionary_name}, target_id={TARGET_ARUCO_ID}")
    print("Keys:")
    print("  Space - pause/play")
    print("  A/D   - back/forward 1 second")
    print("  J/L   - back/forward 5 seconds")
    print("  R     - restart video")
    print("  Q/Esc - quit")


def main() -> int:
    args = parse_args()
    check_aruco_module()

    video_path = choose_video(args.video)
    if video_path is None:
        print("Video was not selected.")
        return 1

    detector = create_detector(args.dict)
    cap, fps, total_frames = open_video(video_path)

    paused = False
    seek_to: int | None = None
    current_frame: np.ndarray | None = None
    current_output: np.ndarray | None = None
    current_mask: np.ndarray | None = None
    trackbar_update = False

    def on_trackbar(value: int) -> None:
        nonlocal seek_to
        if not trackbar_update:
            seek_to = value

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(TRACKBAR, WINDOW, 0, max(1, total_frames - 1), on_trackbar)

    print_help(video_path, args.dict)

    try:
        while True:
            start_time = time.time()

            if seek_to is not None:
                frame_number = max(0, seek_to)
                if total_frames > 0:
                    frame_number = min(frame_number, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, current_frame = cap.read()
                current_output = None
                current_mask = None
                seek_to = None
                paused = True
                if not ok:
                    break
            elif current_frame is None or not paused:
                ok, current_frame = cap.read()
                current_output = None
                current_mask = None
                if not ok:
                    paused = True
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        break
                    if key in (ord("r"), ord("R")):
                        seek_to = 0
                    continue

            frame_index = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)

            if current_output is None and current_frame is not None:
                current_output, current_mask = process_frame(
                    current_frame,
                    frame_index,
                    fps,
                    paused,
                    detector,
                )

            if current_output is not None:
                cv2.imshow(WINDOW, current_output)
            if current_mask is not None:
                cv2.imshow(MASK_WINDOW, current_mask)

            if total_frames > 0:
                trackbar_update = True
                cv2.setTrackbarPos(TRACKBAR, WINDOW, frame_index)
                trackbar_update = False

            delay = 30 if paused else max(1, int(1000 / fps - (time.time() - start_time) * 1000))
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                current_output = None
            elif key in (ord("r"), ord("R")):
                seek_to = 0
            elif key in (ord("a"), ord("A"), ord("d"), ord("D"), ord("j"), ord("J"), ord("l"), ord("L")):
                one_second = max(1, int(round(fps)))
                if key in (ord("a"), ord("A")):
                    seek_to = frame_index - one_second
                elif key in (ord("d"), ord("D")):
                    seek_to = frame_index + one_second
                elif key in (ord("j"), ord("J")):
                    seek_to = frame_index - 5 * one_second
                else:
                    seek_to = frame_index + 5 * one_second
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
