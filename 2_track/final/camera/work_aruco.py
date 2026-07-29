from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DICTIONARY_NAME = "DICT_4X4_50"
DEFAULT_FPS = 30.0
WINDOW = "aruco video viewer"
TRACKBAR = "Frame"

TARGET_ARUCO_ID = 5

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Просмотр видео и поиск ArUco-маркеров.")
    parser.add_argument("video", nargs="?", help="Путь к видео")
    parser.add_argument("--dict", default=DICTIONARY_NAME, help="Словарь ArUco, например DICT_4X4_50")
    return parser.parse_args()


def choose_video(video_arg: str | None) -> Path | None:
    if video_arg:
        return Path(video_arg)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        file_name = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        return Path(file_name) if file_name else None
    except Exception as exc:
        print(f"Не удалось открыть выбор файла: {exc}", file=sys.stderr)
        return None


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Видео не найдено: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(fps) or fps <= 0:
        fps = DEFAULT_FPS

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, max(0, frame_count)


def check_aruco_module() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "В OpenCV отсутствует модуль aruco. "
            "Установите opencv-contrib-python и удалите конфликтующие версии OpenCV."
        )


def create_detector(dictionary_name: str) -> Any:
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"В cv2.aruco нет словаря {dictionary_name}")

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
    else:
        dictionary = cv2.aruco.Dictionary_get(dictionary_id)

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    # Чуть более терпимые настройки для видео.
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


def detect_markers(gray: np.ndarray, detector: Any) -> tuple[Any, Any, Any]:
    if hasattr(detector, "detectMarkers"):
        return detector.detectMarkers(gray)

    dictionary, parameters = detector
    return cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def put_text(frame: np.ndarray, text: str, pos: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def marker_area(corners: np.ndarray) -> float:
    points = corners.reshape(4, 2).astype(np.float32)
    return float(abs(cv2.contourArea(points)))


def draw_marker(
    frame: np.ndarray,
    marker_corners: np.ndarray,
    marker_id: int,
) -> tuple[int, int, int, int, float]:
    points = marker_corners.reshape(4, 2)
    center_x = int(points[:, 0].mean())
    center_y = int(points[:, 1].mean())

    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    error_x = center_x - frame_center_x
    error_y = center_y - frame_center_y
    area = marker_area(marker_corners)

    cv2.circle(frame, (center_x, center_y), 7, (0, 0, 255), -1)
    cv2.line(frame, (frame_center_x, frame_center_y), (center_x, center_y), (255, 0, 0), 2)

    put_text(frame, f"ID: {marker_id}", (center_x + 10, center_y - 10), (0, 255, 0))
    put_text(frame, f"dx={error_x}, dy={error_y}", (20, 70), (0, 255, 0))
    put_text(frame, f"area={area:.0f}", (20, 100), (0, 255, 0))

    return center_x, center_y, error_x, error_y, area


def preprocess(gray: np.ndarray, mode: int) -> tuple[str, np.ndarray]:
    if mode == 0:
        return "gray", gray
    if mode == 1:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return "clahe", clahe.apply(gray)
    if mode == 2:
        blur = cv2.bilateralFilter(gray, 5, 45, 45)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return "bilateral+clahe", clahe.apply(blur)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    return "adaptive", adaptive


def process_frame(
    frame: np.ndarray,
    detector: Any,
    frame_index: int,
    fps: float,
    preprocess_mode: int,
    show_rejected: bool,
) -> np.ndarray:
    output = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mode_name, image_for_detection = preprocess(gray, preprocess_mode)
    corners, ids, rejected = detect_markers(image_for_detection, detector)

    frame_center = (output.shape[1] // 2, output.shape[0] // 2)
    cv2.drawMarker(output, frame_center, (255, 0, 0), cv2.MARKER_CROSS, 25, 2)

    if show_rejected and rejected is not None:
        for rejected_corners in rejected[:100]:
            pts = rejected_corners.reshape(4, 2).astype(np.int32)
            cv2.polylines(output, [pts], True, (0, 220, 255), 1)

    target_corners = None

    if ids is not None:
        for marker_corners, marker_id in zip(
            corners,
            ids.flatten(),
        ):
            if int(marker_id) == TARGET_ARUCO_ID:
                target_corners = marker_corners
                break

    if target_corners is None:
        put_text(
            output,
            f"ARUCO ID {TARGET_ARUCO_ID} NOT FOUND",
            (20, 40),
            (0, 0, 255),
        )
    else:
        target_ids = np.array(
            [[TARGET_ARUCO_ID]],
            dtype=np.int32,
        )

        cv2.aruco.drawDetectedMarkers(
            output,
            [target_corners],
            target_ids,
        )

        draw_marker(
            output,
            target_corners,
            TARGET_ARUCO_ID,
        )

        put_text(
            output,
            "TARGET ID 5 FOUND",
            (20, 40),
            (0, 255, 0),
        )

    seconds = frame_index / fps if fps > 0 else 0.0
    put_text(output, f"frame={frame_index} time={format_time(seconds)}", (20, output.shape[0] - 55), (255, 255, 255))
    put_text(output, f"mode={mode_name} rejected={'ON' if show_rejected else 'OFF'}", (20, output.shape[0] - 25), (255, 255, 255))

    return output


def main() -> int:
    args = parse_args()
    check_aruco_module()

    video_path = choose_video(args.video)
    if video_path is None:
        print("Видео не выбрано.")
        return 1

    detector = create_detector(args.dict)
    cap, fps, total_frames = open_video(video_path)

    paused = False
    show_rejected = False
    preprocess_mode = 0
    seek_to: int | None = None
    current_frame: np.ndarray | None = None
    current_output: np.ndarray | None = None
    trackbar_update = False

    def on_trackbar(value: int) -> None:
        nonlocal seek_to
        if not trackbar_update:
            seek_to = value

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.createTrackbar(TRACKBAR, WINDOW, 0, max(1, total_frames - 1), on_trackbar)

    print(f"Видео: {video_path}")
    print(f"Словарь: {args.dict}")
    print("Клавиши:")
    print("  Space - пауза")
    print("  A/D   - назад/вперёд на 1 секунду")
    print("  J/L   - назад/вперёд на 5 секунд")
    print("  R     - начало видео")
    print("  M     - показать/скрыть rejected candidates")
    print("  P     - сменить предобработку: gray/clahe/bilateral/adaptive")
    print("  Q/Esc - выход")

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
                seek_to = None
                paused = True
                if not ok:
                    break
            elif current_frame is None or not paused:
                ok, current_frame = cap.read()
                current_output = None
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
                current_output = process_frame(
                    current_frame,
                    detector,
                    frame_index,
                    fps,
                    preprocess_mode,
                    show_rejected,
                )

            if current_output is not None:
                cv2.imshow(WINDOW, current_output)

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
            elif key in (ord("m"), ord("M")):
                show_rejected = not show_rejected
                current_output = None
            elif key in (ord("p"), ord("P")):
                preprocess_mode = (preprocess_mode + 1) % 4
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
