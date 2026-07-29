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


DICTIONARY_NAME = "DICT_4X4_50"

# Целевой ArUco-маркер: любые другие декодированные ID игнорируются.
TARGET_ARUCO_ID = 5
TARGET_MARKER_ID = TARGET_ARUCO_ID

DEFAULT_FPS = 30.0
WINDOW = "aruco video viewer"
TRACKBAR = "Frame"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")

# Если True, рядом с исходным видео будет создан файл *_detected с нарисованной разметкой.
SAVE_OUTPUT = False

# Минимальная площадь маркера в пикселях: увеличьте, если сетка дает много мелких ложных кандидатов.
MIN_MARKER_AREA = 1500
# Максимальная доля площади кадра: уменьшите, если крупные блики или края бассейна принимаются за маркер.
MAX_MARKER_AREA_RATIO = 0.75
# Максимальное отношение сторон кандидата: увеличьте, если маркер сильно виден под углом.
MAX_ASPECT_RATIO = 1.8
# Минимальная длина стороны в пикселях: увеличьте для видео с высоким разрешением и крупным маркером.
MIN_SIDE_LENGTH = 25

# Сколько стабильных кадров подряд нужно для подтверждения маркера.
REQUIRED_CONFIRMATIONS = 4
# Максимальный скачок центра между соседними кадрами, при котором трек считается тем же маркером.
MAX_CENTER_SHIFT_PX = 100
# Сколько кадров можно временно не видеть маркер, прежде чем полностью сбросить состояние.
MAX_MISSED_FRAMES = 3
# Коэффициент экспоненциального сглаживания: больше значение - быстрее реакция, меньше - спокойнее картинка.
SMOOTHING_ALPHA = 0.35

REFERENCE_DISTANCE_M = 1.0
REFERENCE_MARKER_SIZE_PX: float | None = None

WARP_SIZE = 240
UPSCALE_FACTOR = 1.5
DEBUG_PANEL_SIZE = (240, 160)

# HSV-порог белого цвета: малая насыщенность и достаточно высокая яркость.
WHITE_SATURATION_MAX = 36
WHITE_VALUE_MIN = 132

# Геометрические ограничения белого прямоугольника относительно площади кадра.
MIN_WHITE_RECT_AREA_RATIO = 0.0327
MAX_WHITE_RECT_AREA_RATIO = 0.5
# Максимальное отношение длинной стороны к короткой: отсекает вытянутые блики и полосы.
MAX_WHITE_RECT_ASPECT_RATIO = 1.31
# Минимальная заполненность minAreaRect контуром: защищает от рваных и пустых прямоугольников.
MIN_RECTANGULARITY = 0.2
# Минимальная сторона прямоугольника в пикселях.
MIN_RECT_SIDE_PX = 1

# Замыкание соединяет белые области, разрезанные черной сеткой.
CLOSE_KERNEL = 3
CLOSE_ITERATIONS = 2

# Открытие удаляет мелкий белый шум после замыкания.
OPEN_KERNEL = 7
OPEN_ITERATIONS = 1

# Размер квадратного изображения для повторного ArUco-детекта внутри контура.
CONTOUR_WARP_SIZE = 500
# Ограничение числа лучших контуров, для которых выполняется дорогой perspective warp.
MAX_CONTOUR_CANDIDATES = 8

# Сколько стабильных кадров требуется, чтобы считать контур подтвержденным.
REQUIRED_CONTOUR_CONFIRMATIONS = 4
# Максимальный допустимый сдвиг центра контура между соседними кадрами.
MAX_CONTOUR_CENTER_SHIFT_PX = 80
# Сколько кадров можно пропустить до сброса истории подтверждения контура.
MAX_CONTOUR_MISSED_FRAMES = 3

PREPROCESS_MODES = (
    "multi",
    "gray",
    "clahe",
    "bilateral_clahe",
    "gaussian_clahe",
    "adaptive_threshold",
)

# Минимальное сходство rejected-кандидата с шаблоном ArUco ID 5.
# Уменьшайте до 0.42..0.48, если сетка сильно закрывает клетки; увеличивайте, если много ложных срабатываний.
TARGET_TEMPLATE_MATCH_MIN_SCORE = 0.50
# Вес сходства с известным ID 5 в общей оценке rejected-кандидата.
TARGET_TEMPLATE_SCORE_WEIGHT = 220.0

TARGET_MARKER_TEMPLATES: list[np.ndarray] = []


@dataclass
class DetectionData:
    found: bool
    decoded: bool
    marker_id: int | None
    corners: np.ndarray | None
    center: tuple[int, int] | None
    area: float
    score: float


@dataclass
class TrackingState:
    confirmations: int = 0
    missed_frames: int = 0
    last_center: tuple[float, float] | None = None
    smoothed_center: np.ndarray | None = None
    smoothed_corners: np.ndarray | None = None
    confirmed: bool = False
    last_decoded_id: int | None = None

    def reset(self) -> None:
        self.confirmations = 0
        self.missed_frames = 0
        self.last_center = None
        self.smoothed_center = None
        self.smoothed_corners = None
        self.confirmed = False
        self.last_decoded_id = None


@dataclass
class ContourTrackingState:
    contour_confirmations: int = 0
    contour_missed_frames: int = 0
    last_contour_center: tuple[float, float] | None = None
    confirmed_contour_center: np.ndarray | None = None
    confirmed_contour_box: np.ndarray | None = None

    def reset(self) -> None:
        self.contour_confirmations = 0
        self.contour_missed_frames = 0
        self.last_contour_center = None
        self.confirmed_contour_center = None
        self.confirmed_contour_box = None

    @property
    def confirmed(self) -> bool:
        return self.contour_confirmations >= REQUIRED_CONTOUR_CONFIRMATIONS


GLOBAL_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Просмотр видео и поиск ArUco-маркера под сеткой.")
    parser.add_argument("video", nargs="?", help="Путь к видеофайлу")
    parser.add_argument("--dict", default=DICTIONARY_NAME, help="Словарь ArUco, например DICT_4X4_50")
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
            title="Выберите видео",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.m4v"),
                ("All files", "*.*"),
            ],
        )
        root.destroy()
        if file_name:
            return Path(file_name)
    except Exception as exc:
        print(f"Не удалось открыть окно выбора файла: {exc}", file=sys.stderr)

    videos = find_video_files()
    if not videos:
        print("Видео не найдено в текущей папке, родительской папке или соседних папках.")
        return None

    if len(videos) == 1:
        print(f"Найдено видео: {videos[0]}")
        return videos[0]

    print("Найдено несколько видео:")
    for index, path in enumerate(videos, start=1):
        print(f"  {index}: {path}")

    while True:
        choice = input("Введите номер видео: ").strip()
        if not choice:
            return None
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(videos):
                return videos[index - 1]
        print("Некорректный номер, попробуйте еще раз.")


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
    if frame_count < 0:
        frame_count = 0
    return cap, fps, frame_count


def check_aruco_module() -> None:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "В OpenCV отсутствует модуль aruco. "
            "Установите opencv-contrib-python и удалите конфликтующие версии OpenCV."
        )


def create_aruco_dictionary(dictionary_name: str) -> Any:
    if not hasattr(cv2.aruco, dictionary_name):
        raise RuntimeError(f"No ArUco dictionary in cv2.aruco: {dictionary_name}")

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def create_target_marker_templates(dictionary_name: str, marker_id: int, size: int = WARP_SIZE) -> list[np.ndarray]:
    dictionary = create_aruco_dictionary(dictionary_name)
    if hasattr(cv2.aruco, "generateImageMarker"):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
    else:
        marker = np.zeros((size, size), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, size, marker, 1)

    _, marker = cv2.threshold(marker, 127, 255, cv2.THRESH_BINARY)
    templates = [marker]
    for rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        templates.append(cv2.rotate(marker, rotate_code))
    return templates


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

    for name, value in {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 65,
        "adaptiveThreshWinSizeStep": 4,
        "adaptiveThreshConstant": 5,
        "minMarkerPerimeterRate": 0.01,
        "maxMarkerPerimeterRate": 2.0,
        "polygonalApproxAccuracyRate": 0.07,
        "minCornerDistanceRate": 0.02,
        "minDistanceToBorder": 2,
        "errorCorrectionRate": 0.7,
    }.items():
        if hasattr(parameters, name):
            setattr(parameters, name, value)

    if hasattr(parameters, "cornerRefinementMethod") and hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)
    return dictionary, parameters


def detect_markers(image: np.ndarray, detector: Any) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
    if hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(image, dictionary, parameters=parameters)

    corner_list = list(corners) if corners is not None else []
    rejected_list = list(rejected) if rejected is not None else []
    return corner_list, ids, rejected_list


def positive_odd_kernel_size(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def normalize_quad(points: Any) -> np.ndarray | None:
    array = np.asarray(points, dtype=np.float32)
    if array.size != 8:
        return None
    array = array.reshape(4, 2)
    if not np.all(np.isfinite(array)):
        return None
    return array


def polygon_area(points: np.ndarray) -> float:
    quad = normalize_quad(points)
    if quad is None:
        return 0.0
    return float(abs(cv2.contourArea(quad)))


def order_points(points: np.ndarray) -> np.ndarray:
    """Возвращает углы: левый верхний, правый верхний, правый нижний, левый нижний."""

    quad = normalize_quad(points)
    if quad is None:
        raise ValueError("Ожидались четыре точки кандидата")

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = quad.sum(axis=1)
    diffs = np.diff(quad, axis=1).reshape(4)
    ordered[0] = quad[np.argmin(sums)]
    ordered[2] = quad[np.argmax(sums)]
    ordered[1] = quad[np.argmin(diffs)]
    ordered[3] = quad[np.argmax(diffs)]

    if len({tuple(point) for point in ordered}) < 4:
        center = quad.mean(axis=0)
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        ordered = quad[np.argsort(angles)]
        top_left_index = np.argmin(ordered.sum(axis=1))
        ordered = np.roll(ordered, -top_left_index, axis=0)
    return ordered.astype(np.float32)


def order_corners(points: np.ndarray) -> np.ndarray:
    return order_points(points)


def warp_candidate(frame: np.ndarray, points: np.ndarray, size: int = WARP_SIZE) -> np.ndarray:
    ordered = order_corners(points)
    destination = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, destination)
    return cv2.warpPerspective(frame, matrix, (size, size))


def create_white_mask(frame: np.ndarray) -> np.ndarray:
    """Создает маску белой области маркера и склеивает разрывы от черной сетки."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, WHITE_VALUE_MIN], dtype=np.uint8)
    upper = np.array([179, WHITE_SATURATION_MAX, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    close_size = positive_odd_kernel_size(CLOSE_KERNEL)
    open_size = positive_odd_kernel_size(OPEN_KERNEL)

    if CLOSE_ITERATIONS > 0:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=CLOSE_ITERATIONS)

    if OPEN_ITERATIONS > 0:
        open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_size, open_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=OPEN_ITERATIONS)

    return mask


def candidate_inside_frame_ratio(points: np.ndarray, frame_shape: tuple[int, ...]) -> float:
    quad = normalize_quad(points)
    if quad is None or len(frame_shape) < 2:
        return 0.0

    height, width = frame_shape[:2]
    x, y, box_width, box_height = cv2.boundingRect(quad.astype(np.float32))
    if box_width <= 0 or box_height <= 0:
        return 0.0

    x2 = x + box_width
    y2 = y + box_height
    inside_width = max(0, min(x2, width) - max(x, 0))
    inside_height = max(0, min(y2, height) - max(y, 0))
    return (inside_width * inside_height) / float(box_width * box_height)


def find_white_rectangle_candidates(frame: np.ndarray, white_mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = frame.shape[:2]
    frame_area = float(height * width)
    if frame_area <= 0:
        return []

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[dict[str, Any]] = []

    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        if contour_area <= 0:
            continue

        area_ratio = contour_area / frame_area
        if area_ratio < MIN_WHITE_RECT_AREA_RATIO or area_ratio > MAX_WHITE_RECT_AREA_RATIO:
            continue

        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = float(rect[1][0]), float(rect[1][1])
        if rect_width <= 0 or rect_height <= 0:
            continue

        min_side = min(rect_width, rect_height)
        max_side = max(rect_width, rect_height)
        if min_side < MIN_RECT_SIDE_PX:
            continue

        aspect_ratio = max_side / max(1.0, min_side)
        if aspect_ratio > MAX_WHITE_RECT_ASPECT_RATIO:
            continue

        rectangle_area = rect_width * rect_height
        if rectangle_area <= 0:
            continue

        rectangularity = contour_area / rectangle_area
        if rectangularity < MIN_RECTANGULARITY:
            continue

        box = order_points(cv2.boxPoints(rect))
        if candidate_inside_frame_ratio(box, frame.shape) < 0.75:
            continue

        center = tuple(np.round(box.mean(axis=0)).astype(int))
        square_bonus = 1.0 / max(aspect_ratio, 1.0)
        area_bonus = min(1.0, area_ratio / max(MIN_WHITE_RECT_AREA_RATIO, 1e-6))
        score = area_ratio * rectangularity * square_bonus * (1.0 + 0.25 * area_bonus)

        candidates.append(
            {
                "contour": contour,
                "contour_area": contour_area,
                "area_ratio": area_ratio,
                "rect": rect,
                "width": rect_width,
                "height": rect_height,
                "aspect_ratio": aspect_ratio,
                "rectangularity": rectangularity,
                "box": box,
                "center": (int(center[0]), int(center[1])),
                "score": float(score),
            }
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates


def expand_candidate_points(points: np.ndarray, frame_shape: tuple[int, ...], scale: float = 1.08) -> np.ndarray:
    quad = order_points(points)
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * scale

    height, width = frame_shape[:2]
    expanded[:, 0] = np.clip(expanded[:, 0], 0, max(0, width - 1))
    expanded[:, 1] = np.clip(expanded[:, 1], 0, max(0, height - 1))
    return expanded.astype(np.float32)


def warp_contour_candidate(frame: np.ndarray, candidate_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expanded = expand_candidate_points(candidate_points, frame.shape)
    destination = np.array(
        [
            [0, 0],
            [CONTOUR_WARP_SIZE - 1, 0],
            [CONTOUR_WARP_SIZE - 1, CONTOUR_WARP_SIZE - 1],
            [0, CONTOUR_WARP_SIZE - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(expanded, destination)
    if not np.all(np.isfinite(matrix)) or abs(float(np.linalg.det(matrix))) < 1e-9:
        raise ValueError("Вырожденная матрица перспективного преобразования")
    warp = cv2.warpPerspective(frame, matrix, (CONTOUR_WARP_SIZE, CONTOUR_WARP_SIZE))
    return warp, matrix


def side_lengths(points: np.ndarray) -> np.ndarray:
    ordered = order_corners(points)
    return np.array(
        [
            np.linalg.norm(ordered[1] - ordered[0]),
            np.linalg.norm(ordered[2] - ordered[1]),
            np.linalg.norm(ordered[3] - ordered[2]),
            np.linalg.norm(ordered[0] - ordered[3]),
        ],
        dtype=np.float32,
    )


def candidate_geometry_is_valid(points: np.ndarray, frame_shape: tuple[int, ...]) -> bool:
    quad = normalize_quad(points)
    if quad is None or len(frame_shape) < 2:
        return False

    height, width = frame_shape[:2]
    if height <= 0 or width <= 0:
        return False

    ordered = order_corners(quad)
    contour = ordered.reshape(-1, 1, 2).astype(np.float32)
    if not cv2.isContourConvex(contour):
        return False

    area = polygon_area(ordered)
    frame_area = float(height * width)
    if area < MIN_MARKER_AREA or area > frame_area * MAX_MARKER_AREA_RATIO:
        return False

    lengths = side_lengths(ordered)
    if np.min(lengths) < MIN_SIDE_LENGTH:
        return False

    marker_width = float((lengths[0] + lengths[2]) * 0.5)
    marker_height = float((lengths[1] + lengths[3]) * 0.5)
    if marker_width <= 0 or marker_height <= 0:
        return False
    aspect_ratio = max(marker_width, marker_height) / min(marker_width, marker_height)
    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    opposite_width_ratio = max(lengths[0], lengths[2]) / max(1.0, min(lengths[0], lengths[2]))
    opposite_height_ratio = max(lengths[1], lengths[3]) / max(1.0, min(lengths[1], lengths[3]))
    if opposite_width_ratio > 2.2 or opposite_height_ratio > 2.2:
        return False

    x, y, box_width, box_height = cv2.boundingRect(ordered.astype(np.float32))
    if box_width <= 0 or box_height <= 0:
        return False
    x2 = x + box_width
    y2 = y + box_height
    inside_width = max(0, min(x2, width) - max(x, 0))
    inside_height = max(0, min(y2, height) - max(y, 0))
    inside_ratio = (inside_width * inside_height) / float(box_width * box_height)
    return inside_ratio >= 0.75


def candidate_content_metrics(frame: np.ndarray, points: np.ndarray) -> dict[str, float] | None:
    if not candidate_geometry_is_valid(points, frame.shape):
        return None

    warp = warp_candidate(frame, points, WARP_SIZE)
    if warp.size == 0:
        return None

    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    hue = hsv[:, :, 0]

    dark_mask = gray < 85
    bright_mask = gray > 165
    blue_mask = (hue >= 85) & (hue <= 135) & (saturation > 35) & (value > 45)

    dark_ratio = float(np.mean(dark_mask))
    bright_ratio = float(np.mean(bright_mask))
    blue_ratio = float(np.mean(blue_mask))
    contrast = float(np.percentile(gray, 90) - np.percentile(gray, 10))

    border = max(12, WARP_SIZE // 9)
    border_mask = np.zeros(gray.shape, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    border_dark_ratio = float(np.mean(dark_mask[border_mask]))

    inner = gray[border:-border, border:-border]
    if inner.size == 0:
        return None
    _, bright_inner = cv2.threshold(inner, 165, 255, cv2.THRESH_BINARY)
    bright_inner = cv2.morphologyEx(
        bright_inner,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(bright_inner, 8)
    min_component_area = max(20, int(inner.size * 0.004))
    large_bright_components = 0
    for index in range(1, component_count):
        if int(stats[index, cv2.CC_STAT_AREA]) >= min_component_area:
            large_bright_components += 1

    if contrast < 35:
        return None
    if dark_ratio < 0.07 or bright_ratio < 0.05:
        return None
    if dark_ratio > 0.88 or bright_ratio > 0.88:
        return None
    if border_dark_ratio < 0.20:
        return None
    if large_bright_components < 2:
        return None
    if blue_ratio > 0.60:
        return None

    return {
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "border_dark_ratio": border_dark_ratio,
        "contrast": contrast,
        "blue_ratio": blue_ratio,
        "bright_components": float(large_bright_components),
        "area": polygon_area(points),
    }


def cell_pattern_score(candidate_binary: np.ndarray, template_binary: np.ndarray, cells: int = 6) -> float:
    height, width = candidate_binary.shape[:2]
    cell_h = height / cells
    cell_w = width / cells
    matches = 0
    total = 0
    for row in range(cells):
        for col in range(cells):
            y1 = int(round((row + 0.25) * cell_h))
            y2 = int(round((row + 0.75) * cell_h))
            x1 = int(round((col + 0.25) * cell_w))
            x2 = int(round((col + 0.75) * cell_w))
            if y2 <= y1 or x2 <= x1:
                continue
            candidate_is_bright = float(np.mean(candidate_binary[y1:y2, x1:x2])) > 127.0
            template_is_bright = float(np.mean(template_binary[y1:y2, x1:x2])) > 127.0
            matches += int(candidate_is_bright == template_is_bright)
            total += 1
    return matches / total if total else 0.0


def target_marker_template_score(frame: np.ndarray, points: np.ndarray) -> float | None:
    if not TARGET_MARKER_TEMPLATES:
        return None

    warp = warp_candidate(frame, points, WARP_SIZE)
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, candidate = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    best_score = 0.0
    for template in TARGET_MARKER_TEMPLATES:
        pixel_score = float(np.mean(candidate == template))
        inverted_pixel_score = float(np.mean((255 - candidate) == template))
        cell_score = cell_pattern_score(candidate, template)
        inverted_cell_score = cell_pattern_score(255 - candidate, template)
        score = max(
            pixel_score * 0.35 + cell_score * 0.65,
            inverted_pixel_score * 0.25 + inverted_cell_score * 0.55,
        )
        best_score = max(best_score, score)

    return best_score


def score_rejected_candidate(frame: np.ndarray, points: np.ndarray) -> float | None:
    metrics = candidate_content_metrics(frame, points)
    if metrics is None:
        return None

    template_score = target_marker_template_score(frame, points)
    if template_score is None or template_score < TARGET_TEMPLATE_MATCH_MIN_SCORE:
        return None

    balance = 1.0 - abs(metrics["dark_ratio"] - metrics["bright_ratio"])
    area_bonus = min(45.0, math.sqrt(metrics["area"]) * 0.35)
    score = (
        metrics["contrast"] * 1.15
        + metrics["border_dark_ratio"] * 95.0
        + min(metrics["bright_components"], 8.0) * 12.0
        + balance * 35.0
        + area_bonus
        + template_score * TARGET_TEMPLATE_SCORE_WEIGHT
        - metrics["blue_ratio"] * 90.0
    )
    return float(score)


def make_detection_data(
    found: bool,
    decoded: bool = False,
    marker_id: int | None = None,
    corners: np.ndarray | None = None,
    score: float = 0.0,
) -> DetectionData:
    if corners is None:
        return DetectionData(False, False, None, None, None, 0.0, 0.0)

    quad = order_corners(corners)
    center = tuple(np.round(quad.mean(axis=0)).astype(int))
    return DetectionData(found, decoded, marker_id, quad, center, polygon_area(quad), float(score))


def find_best_marker(
    frame: np.ndarray,
    corners: list[np.ndarray],
    ids: np.ndarray | None,
    rejected: list[np.ndarray],
) -> dict[str, Any]:
    del frame, rejected

    decoded_markers: list[tuple[float, np.ndarray, int]] = []
    if ids is not None and len(corners) > 0:
        flat_ids = np.asarray(ids).reshape(-1)
        for marker_corners, marker_id in zip(corners, flat_ids):
            if int(marker_id) != TARGET_ARUCO_ID:
                continue
            quad = normalize_quad(marker_corners)
            if quad is None:
                continue
            area = polygon_area(quad)
            if area > 0:
                decoded_markers.append((area, quad, int(marker_id)))

    if decoded_markers:
        area, quad, marker_id = max(decoded_markers, key=lambda item: item[0])
        detection = make_detection_data(True, True, marker_id, quad, 10000.0 + area)
        return detection.__dict__ | {"id": detection.marker_id}

    return {
        "found": False,
        "decoded": False,
        "id": None,
        "corners": None,
        "center": None,
        "area": 0.0,
        "score": 0.0,
    }


def prefer_candidate_near_previous(
    frame: np.ndarray,
    detection: dict[str, Any],
    rejected: list[np.ndarray],
    state: TrackingState,
) -> dict[str, Any]:
    if detection.get("decoded") or state.last_center is None:
        return detection

    previous = np.asarray(state.last_center, dtype=np.float32)
    current_far = True
    if detection.get("found") and detection.get("center") is not None:
        current_center = np.asarray(detection["center"], dtype=np.float32)
        current_far = float(np.linalg.norm(current_center - previous)) > MAX_CENTER_SHIFT_PX

    best_quad: np.ndarray | None = None
    best_score: float | None = None
    for candidate in rejected:
        quad = normalize_quad(candidate)
        if quad is None:
            continue
        center = quad.mean(axis=0)
        if float(np.linalg.norm(center - previous)) > MAX_CENTER_SHIFT_PX:
            continue
        score = score_rejected_candidate(frame, quad)
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_quad = quad
            best_score = score

    if best_quad is None or best_score is None:
        return detection
    if detection.get("found") and not current_far and best_score < float(detection.get("score", 0.0)) * 0.55:
        return detection

    near_detection = make_detection_data(True, False, TARGET_MARKER_ID, best_quad, best_score)
    return near_detection.__dict__ | {"id": TARGET_MARKER_ID}


def rescale_quads(quads: list[np.ndarray], scale: float) -> list[np.ndarray]:
    if scale == 1.0:
        return quads
    result: list[np.ndarray] = []
    for quad in quads:
        normalized = normalize_quad(quad)
        if normalized is not None:
            result.append(normalized / scale)
    return result


def build_preprocess_variants(image: np.ndarray, clahe: cv2.CLAHE) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    clahe_gray = clahe.apply(gray)
    bilateral = cv2.bilateralFilter(gray, 7, 50, 50)
    gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive = cv2.adaptiveThreshold(
        clahe_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )
    return {
        "gray": gray,
        "clahe": clahe_gray,
        "bilateral_clahe": clahe.apply(bilateral),
        "gaussian_clahe": clahe.apply(gaussian),
        "adaptive_threshold": adaptive,
    }


def detect_target_id(image: np.ndarray, detector: Any) -> np.ndarray | None:
    """Ищет только TARGET_ARUCO_ID и игнорирует все остальные декодированные ID."""

    variants = build_preprocess_variants(image, GLOBAL_CLAHE)
    for variant in (
        variants["gray"],
        variants["clahe"],
        variants["bilateral_clahe"],
        variants["gaussian_clahe"],
        variants["adaptive_threshold"],
    ):
        corners, ids, _ = detect_markers(variant, detector)
        if ids is None or not corners:
            continue
        for marker_corners, marker_id in zip(corners, np.asarray(ids).reshape(-1)):
            if int(marker_id) != TARGET_ARUCO_ID:
                continue
            quad = normalize_quad(marker_corners)
            if quad is not None and polygon_area(quad) > 0:
                return order_points(quad)
    return None


def empty_contour_result(white_mask: np.ndarray | None = None) -> dict[str, Any]:
    return {
        "found": False,
        "decoded": False,
        "marker_corners": None,
        "candidate_box": None,
        "candidate_center": None,
        "white_mask": white_mask if white_mask is not None else np.zeros((1, 1), dtype=np.uint8),
        "score": 0.0,
        "area_ratio": 0.0,
        "aspect_ratio": 0.0,
        "rectangularity": 0.0,
        "candidates": [],
        "best_warp": None,
        "best_warp_threshold": None,
    }


def detect_target_via_contours(frame: np.ndarray, detector: Any) -> dict[str, Any]:
    white_mask = create_white_mask(frame)
    candidates = find_white_rectangle_candidates(frame, white_mask)
    if not candidates:
        return empty_contour_result(white_mask)

    best_candidate = candidates[0]
    best_warp: np.ndarray | None = None
    best_threshold: np.ndarray | None = None

    for candidate in candidates[:MAX_CONTOUR_CANDIDATES]:
        try:
            warp, matrix = warp_contour_candidate(frame, candidate["box"])
        except (cv2.error, ValueError):
            continue

        if best_warp is None:
            best_warp = warp
            warp_variants = build_preprocess_variants(warp, GLOBAL_CLAHE)
            best_threshold = warp_variants["adaptive_threshold"]

        marker_corners = detect_target_id(warp, detector)
        if marker_corners is None:
            continue

        try:
            inverse_matrix = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            continue

        source_corners = cv2.perspectiveTransform(marker_corners.reshape(1, 4, 2), inverse_matrix).reshape(4, 2)
        source_corners = order_points(source_corners)
        return {
            "found": True,
            "decoded": True,
            "marker_corners": source_corners,
            "candidate_box": candidate["box"],
            "candidate_center": candidate["center"],
            "white_mask": white_mask,
            "score": float(candidate["score"]),
            "area_ratio": float(candidate["area_ratio"]),
            "aspect_ratio": float(candidate["aspect_ratio"]),
            "rectangularity": float(candidate["rectangularity"]),
            "candidates": candidates,
            "best_warp": warp,
            "best_warp_threshold": build_preprocess_variants(warp, GLOBAL_CLAHE)["adaptive_threshold"],
        }

    return {
        "found": True,
        "decoded": False,
        "marker_corners": None,
        "candidate_box": best_candidate["box"],
        "candidate_center": best_candidate["center"],
        "white_mask": white_mask,
        "score": float(best_candidate["score"]),
        "area_ratio": float(best_candidate["area_ratio"]),
        "aspect_ratio": float(best_candidate["aspect_ratio"]),
        "rectangularity": float(best_candidate["rectangularity"]),
        "candidates": candidates,
        "best_warp": best_warp,
        "best_warp_threshold": best_threshold,
    }


def run_multivariant_detection(
    frame: np.ndarray,
    detector: Any,
    clahe: cv2.CLAHE,
    preprocessing_mode: str = "multi",
) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray], dict[str, np.ndarray]]:
    debug_images = build_preprocess_variants(frame, clahe)
    if preprocessing_mode == "multi":
        variants: list[tuple[str, np.ndarray, float]] = [
            ("gray", debug_images["gray"], 1.0),
            ("clahe", debug_images["clahe"], 1.0),
            ("bilateral_clahe", debug_images["bilateral_clahe"], 1.0),
            ("gaussian_clahe", debug_images["gaussian_clahe"], 1.0),
            ("adaptive_threshold", debug_images["adaptive_threshold"], 1.0),
        ]
    else:
        variants = [(preprocessing_mode, debug_images.get(preprocessing_mode, debug_images["gray"]), 1.0)]

    all_corners: list[np.ndarray] = []
    all_ids: list[int] = []
    all_rejected: list[np.ndarray] = []

    for _, image, scale in variants:
        corners, ids, rejected = detect_markers(image, detector)
        scaled_corners = rescale_quads(corners, scale)
        all_corners.extend(scaled_corners)
        if ids is not None and scaled_corners:
            all_ids.extend(int(item) for item in np.asarray(ids).reshape(-1)[: len(scaled_corners)])
        all_rejected.extend(rescale_quads(rejected, scale))

    ids_array = np.asarray(all_ids, dtype=np.int32).reshape(-1, 1) if all_ids else None
    return all_corners[: len(all_ids)] if all_ids else all_corners, ids_array, all_rejected, debug_images


def create_grid_mask(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    min_length = max(40, int(min(gray.shape[:2]) * 0.12))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 45, minLineLength=min_length, maxLineGap=16)
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = math.hypot(float(x2 - x1), float(y2 - y1))
            if length >= min_length:
                cv2.line(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, 2)
    return mask


def update_tracking(state: TrackingState, detection: dict[str, Any]) -> str:
    if not detection["found"] or detection["center"] is None or detection["corners"] is None:
        state.missed_frames += 1
        if state.confirmed and state.missed_frames <= MAX_MISSED_FRAMES and state.smoothed_center is not None:
            return "MARKER PREDICTED"
        if state.missed_frames > MAX_MISSED_FRAMES:
            state.reset()
        return "MARKER NOT FOUND"

    center = np.asarray(detection["center"], dtype=np.float32)
    corners = np.asarray(detection["corners"], dtype=np.float32)
    stable = False
    if state.last_center is not None:
        previous = np.asarray(state.last_center, dtype=np.float32)
        stable = float(np.linalg.norm(center - previous)) <= MAX_CENTER_SHIFT_PX

    if state.last_center is None:
        state.confirmations = 1
        state.smoothed_center = center
        state.smoothed_corners = corners
    elif stable:
        state.confirmations += 1
        if state.smoothed_center is None or state.smoothed_corners is None:
            state.smoothed_center = center
            state.smoothed_corners = corners
        else:
            state.smoothed_center = (1.0 - SMOOTHING_ALPHA) * state.smoothed_center + SMOOTHING_ALPHA * center
            state.smoothed_corners = (1.0 - SMOOTHING_ALPHA) * state.smoothed_corners + SMOOTHING_ALPHA * corners
    else:
        state.confirmations = 1
        state.smoothed_center = center
        state.smoothed_corners = corners

    state.missed_frames = 0
    state.last_center = (float(center[0]), float(center[1]))
    if detection["decoded"] and detection["id"] is not None:
        state.last_decoded_id = int(detection["id"])
    state.confirmed = state.confirmations >= REQUIRED_CONFIRMATIONS

    if state.confirmed:
        return "MARKER CONFIRMED"
    return "ARUCO DECODED" if detection["decoded"] else "MARKER CANDIDATE"


def update_contour_tracking(state: ContourTrackingState, contour_result: dict[str, Any]) -> None:
    center_value = contour_result.get("candidate_center")
    box_value = contour_result.get("candidate_box")
    if not contour_result.get("found") or center_value is None or box_value is None:
        state.contour_missed_frames += 1
        if state.contour_missed_frames > MAX_CONTOUR_MISSED_FRAMES:
            state.reset()
        return

    center = np.asarray(center_value, dtype=np.float32)
    box = order_points(np.asarray(box_value, dtype=np.float32))

    stable = False
    if state.last_contour_center is not None:
        previous = np.asarray(state.last_contour_center, dtype=np.float32)
        stable = float(np.linalg.norm(center - previous)) <= MAX_CONTOUR_CENTER_SHIFT_PX

    if state.last_contour_center is None or not stable:
        state.contour_confirmations = 1
        state.confirmed_contour_center = center
        state.confirmed_contour_box = box
    else:
        state.contour_confirmations += 1
        if state.confirmed_contour_center is None or state.confirmed_contour_box is None:
            state.confirmed_contour_center = center
            state.confirmed_contour_box = box
        else:
            state.confirmed_contour_center = (
                (1.0 - SMOOTHING_ALPHA) * state.confirmed_contour_center + SMOOTHING_ALPHA * center
            )
            state.confirmed_contour_box = (
                (1.0 - SMOOTHING_ALPHA) * state.confirmed_contour_box + SMOOTHING_ALPHA * box
            )

    state.contour_missed_frames = 0
    state.last_contour_center = (float(center[0]), float(center[1]))


def format_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    minutes = int(seconds // 60)
    rest = seconds - minutes * 60
    return f"{minutes:02d}:{rest:05.2f}"


def put_text(frame: np.ndarray, text: str, pos: tuple[int, int], color: tuple[int, int, int], scale: float = 0.62) -> None:
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def marker_size_px(points: np.ndarray) -> float:
    lengths = side_lengths(points)
    return float(np.mean(lengths))


def distance_text(size_px: float) -> str:
    if REFERENCE_MARKER_SIZE_PX is None or size_px <= 0:
        return "DISTANCE: calibration required"
    distance = REFERENCE_DISTANCE_M * REFERENCE_MARKER_SIZE_PX / size_px
    return f"DISTANCE: {distance:.2f} m"


def draw_center_cross(frame: np.ndarray) -> tuple[int, int]:
    center = (frame.shape[1] // 2, frame.shape[0] // 2)
    cv2.drawMarker(frame, center, (255, 0, 0), cv2.MARKER_CROSS, 28, 2)
    return center


def draw_marker_overlay(
    frame: np.ndarray,
    detection: dict[str, Any],
    contour_state: ContourTrackingState,
    status: str,
    preprocessing_mode: str,
) -> None:
    frame_center = draw_center_cross(frame)
    candidate_box = detection.get("candidate_box")
    decoded_corners = detection.get("corners")
    contour_box = None
    center_array: np.ndarray | None = None
    display_points: np.ndarray | None = None

    if candidate_box is not None:
        candidate_points = order_points(np.asarray(candidate_box, dtype=np.float32))
        cv2.polylines(frame, [np.round(candidate_points).astype(np.int32)], True, (0, 255, 255), 2)

    if contour_state.confirmed and contour_state.confirmed_contour_box is not None:
        contour_box = order_points(contour_state.confirmed_contour_box)
        cv2.polylines(frame, [np.round(contour_box).astype(np.int32)], True, (0, 165, 255), 5)

    if bool(detection.get("decoded")) and decoded_corners is not None:
        display_points = order_points(np.asarray(decoded_corners, dtype=np.float32))
        center_array = display_points.mean(axis=0)
        cv2.polylines(frame, [np.round(display_points).astype(np.int32)], True, (0, 255, 0), 3)
    elif contour_state.confirmed and contour_box is not None:
        display_points = contour_box
        center_array = contour_state.confirmed_contour_center
    elif candidate_box is not None:
        display_points = order_points(np.asarray(candidate_box, dtype=np.float32))
        center_array = display_points.mean(axis=0)

    status_color = (0, 255, 0) if bool(detection.get("decoded")) else (0, 165, 255)
    if status == "WHITE RECTANGLE FOUND, ID NOT DECODED":
        status_color = (0, 255, 255)
    if status == "ARUCO ID 5 NOT FOUND":
        status_color = (0, 0, 255)

    dx_text = "dx=? dy=?"
    area = float(detection.get("area", 0.0))
    size_px = 0.0
    if display_points is not None:
        if area <= 0:
            area = polygon_area(display_points)
        size_px = marker_size_px(display_points)

    if center_array is not None:
        center = tuple(np.round(center_array).astype(int))
        cv2.circle(frame, center, 7, (0, 0, 255), -1)
        cv2.line(frame, frame_center, center, (255, 0, 0), 2)
        dx = center[0] - frame_center[0]
        dy = center[1] - frame_center[1]
        dx_text = f"dx={dx} dy={dy}"
        if detection.get("decoded"):
            put_text(frame, f"ID: {TARGET_ARUCO_ID}", (center[0] + 10, max(25, center[1] - 10)), (0, 255, 0))

    put_text(frame, status, (20, 40), status_color)
    if status.startswith("CONTOUR CONFIRMED"):
        put_text(frame, "ID 5 NOT DECODED", (20, 72), (0, 165, 255))
        text_y = 104
    else:
        text_y = 72

    put_text(frame, dx_text, (20, text_y), (255, 255, 255))
    put_text(
        frame,
        f"area={area:.0f} area_ratio={float(detection.get('area_ratio', 0.0)):.4f}",
        (20, text_y + 32),
        (255, 255, 255),
    )
    put_text(
        frame,
        f"aspect={float(detection.get('aspect_ratio', 0.0)):.2f} rect={float(detection.get('rectangularity', 0.0)):.2f}",
        (20, text_y + 64),
        (255, 255, 255),
    )
    put_text(frame, f"score={float(detection.get('score', 0.0)):.5f} size={size_px:.1f}px", (20, text_y + 96), (255, 255, 255))
    put_text(
        frame,
        f"CONTOUR CONFIRM: {min(contour_state.contour_confirmations, REQUIRED_CONTOUR_CONFIRMATIONS)}/{REQUIRED_CONTOUR_CONFIRMATIONS}",
        (20, text_y + 128),
        (255, 255, 255),
    )
    put_text(frame, f"PREPROCESS: {preprocessing_mode}", (20, text_y + 160), (255, 255, 255))


def draw_rejected_panel(frame: np.ndarray, rejected: list[np.ndarray], best: np.ndarray | None) -> np.ndarray:
    panel = frame.copy()
    for candidate in rejected[:140]:
        quad = normalize_quad(candidate)
        if quad is not None:
            cv2.polylines(panel, [np.round(quad).astype(np.int32)], True, (0, 180, 255), 1)
    if best is not None:
        cv2.polylines(panel, [np.round(best).astype(np.int32)], True, (0, 255, 255), 3)
    return panel


def candidate_binary_panel(frame: np.ndarray, points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)
    warp = warp_candidate(frame, points, WARP_SIZE)
    gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
    dark = gray < 85
    bright = gray > 165
    panel = np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)
    panel[dark] = (80, 80, 80)
    panel[bright] = (255, 255, 255)
    return panel


def prepare_debug_panel(
    frame: np.ndarray,
    debug_images: dict[str, np.ndarray],
    rejected: list[np.ndarray],
    best_points: np.ndarray | None,
) -> np.ndarray:
    grid_mask = create_grid_mask(frame)
    panels: list[tuple[str, np.ndarray]] = [
        ("grayscale", debug_images.get("gray", np.zeros(frame.shape[:2], dtype=np.uint8))),
        ("CLAHE", debug_images.get("clahe", np.zeros(frame.shape[:2], dtype=np.uint8))),
        ("rejected", draw_rejected_panel(frame, rejected, best_points)),
        ("best warp", warp_candidate(frame, best_points, WARP_SIZE) if best_points is not None else np.zeros((WARP_SIZE, WARP_SIZE, 3), dtype=np.uint8)),
        ("light/dark", candidate_binary_panel(frame, best_points)),
        ("grid mask", grid_mask),
    ]

    rendered = []
    for title, image in panels:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(image, DEBUG_PANEL_SIZE, interpolation=cv2.INTER_AREA)
        put_text(tile, title, (8, 24), (255, 255, 255), 0.55)
        rendered.append(tile)

    top = np.hstack(rendered[:3])
    bottom = np.hstack(rendered[3:])
    return np.vstack([top, bottom])


def draw_contour_candidates_panel(frame: np.ndarray, candidates: list[dict[str, Any]]) -> np.ndarray:
    panel = frame.copy()
    for index, candidate in enumerate(candidates[:MAX_CONTOUR_CANDIDATES]):
        color = (0, 255, 255) if index == 0 else (0, 180, 255)
        box = normalize_quad(candidate.get("box"))
        if box is not None:
            cv2.polylines(panel, [np.round(box).astype(np.int32)], True, color, 2 if index == 0 else 1)
    return panel


def prepare_contour_debug_panel(frame: np.ndarray, contour_result: dict[str, Any]) -> np.ndarray:
    white_mask = contour_result.get("white_mask")
    if white_mask is None:
        white_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    best_warp = contour_result.get("best_warp")
    if best_warp is None:
        best_warp = np.zeros((CONTOUR_WARP_SIZE, CONTOUR_WARP_SIZE, 3), dtype=np.uint8)

    best_threshold = contour_result.get("best_warp_threshold")
    if best_threshold is None:
        best_threshold = np.zeros((CONTOUR_WARP_SIZE, CONTOUR_WARP_SIZE), dtype=np.uint8)

    panels: list[tuple[str, np.ndarray]] = [
        ("source", frame),
        ("white_mask", white_mask),
        ("contours", draw_contour_candidates_panel(frame, contour_result.get("candidates", []))),
        ("candidate warp", best_warp),
        ("warp adaptive", best_threshold),
    ]

    rendered = []
    for title, image in panels:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        tile = cv2.resize(image, DEBUG_PANEL_SIZE, interpolation=cv2.INTER_AREA)
        put_text(tile, title, (8, 24), (255, 255, 255), 0.55)
        rendered.append(tile)

    while len(rendered) < 6:
        rendered.append(np.zeros((DEBUG_PANEL_SIZE[1], DEBUG_PANEL_SIZE[0], 3), dtype=np.uint8))

    top = np.hstack(rendered[:3])
    bottom = np.hstack(rendered[3:6])
    return np.vstack([top, bottom])


def compose_display(output: np.ndarray, debug_panel: np.ndarray | None) -> np.ndarray:
    if debug_panel is None:
        return output
    target_height = min(output.shape[0], debug_panel.shape[0])
    main_width = int(output.shape[1] * target_height / output.shape[0])
    panel_width = int(debug_panel.shape[1] * target_height / debug_panel.shape[0])
    main = cv2.resize(output, (main_width, target_height), interpolation=cv2.INTER_AREA)
    panel = cv2.resize(debug_panel, (panel_width, target_height), interpolation=cv2.INTER_AREA)
    return np.hstack([main, panel])


def draw_frame_info(
    output: np.ndarray,
    frame_index: int,
    total_frames: int,
    fps: float,
    paused: bool,
) -> None:
    current_frame_number = frame_index + 1
    current_seconds = frame_index / fps if fps > 0 else 0.0
    total_seconds = total_frames / fps if total_frames > 0 and fps > 0 else 0.0
    total_text = str(total_frames) if total_frames > 0 else "?"
    duration_text = format_time(total_seconds) if total_frames > 0 else "?:??"
    put_text(
        output,
        f"FRAME: {current_frame_number}/{total_text}  TIME: {format_time(current_seconds)}/{duration_text}",
        (20, output.shape[0] - 55),
        (255, 255, 255),
    )
    if paused:
        put_text(output, "PAUSE", (20, output.shape[0] - 25), (0, 255, 255))


def process_frame(
    frame: np.ndarray,
    detector: Any,
    clahe: cv2.CLAHE,
    contour_state: ContourTrackingState,
    frame_index: int,
    total_frames: int,
    fps: float,
    paused: bool,
    show_rejected_debug: bool,
    show_contour_debug: bool,
    preprocessing_mode: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    output = frame.copy()
    corners, ids, rejected, debug_images = run_multivariant_detection(frame, detector, clahe, preprocessing_mode)
    detection = find_best_marker(frame, corners, ids, rejected)

    contour_result = empty_contour_result(np.zeros(frame.shape[:2], dtype=np.uint8))
    frame_area = float(frame.shape[0] * frame.shape[1])
    if detection["found"] and detection["decoded"]:
        contour_state.reset()
        status = "ARUCO ID 5 FOUND"
        points = order_points(np.asarray(detection["corners"], dtype=np.float32))
        area = polygon_area(points)
        lengths = side_lengths(points)
        aspect_ratio = float(max(lengths) / max(1.0, min(lengths))) if len(lengths) else 0.0
        detection.update(
            {
                "corners": points,
                "candidate_box": None,
                "area": area,
                "area_ratio": area / frame_area if frame_area > 0 else 0.0,
                "aspect_ratio": aspect_ratio,
                "rectangularity": 1.0,
                "score": float(detection.get("score", 0.0)),
            }
        )
    else:
        contour_result = detect_target_via_contours(frame, detector)
        update_contour_tracking(contour_state, contour_result)

        if contour_result["found"] and contour_result["decoded"]:
            status = "ARUCO ID 5 FOUND VIA CONTOUR"
            points = order_points(np.asarray(contour_result["marker_corners"], dtype=np.float32))
            area = polygon_area(points)
            detection = {
                "found": True,
                "decoded": True,
                "id": TARGET_ARUCO_ID,
                "corners": points,
                "center": tuple(np.round(points.mean(axis=0)).astype(int)),
                "area": area,
                "score": contour_result["score"],
                "candidate_box": contour_result["candidate_box"],
                "area_ratio": contour_result["area_ratio"],
                "aspect_ratio": contour_result["aspect_ratio"],
                "rectangularity": contour_result["rectangularity"],
            }
        elif contour_result["found"]:
            candidate_box = order_points(np.asarray(contour_result["candidate_box"], dtype=np.float32))
            if contour_state.confirmed and contour_state.confirmed_contour_box is not None:
                status = "CONTOUR CONFIRMED, ID 5 NOT DECODED"
                box_for_area = order_points(contour_state.confirmed_contour_box)
                center = tuple(np.round(box_for_area.mean(axis=0)).astype(int))
                area = polygon_area(box_for_area)
                corners_for_display: np.ndarray | None = box_for_area
            else:
                status = "WHITE RECTANGLE FOUND, ID NOT DECODED"
                center = contour_result["candidate_center"]
                area = polygon_area(candidate_box)
                corners_for_display = None

            detection = {
                "found": True,
                "decoded": False,
                "id": None,
                "corners": corners_for_display,
                "center": center,
                "area": area,
                "score": contour_result["score"],
                "candidate_box": candidate_box,
                "area_ratio": contour_result["area_ratio"],
                "aspect_ratio": contour_result["aspect_ratio"],
                "rectangularity": contour_result["rectangularity"],
            }
        else:
            status = "ARUCO ID 5 NOT FOUND"
            detection = {
                "found": False,
                "decoded": False,
                "id": None,
                "corners": None,
                "center": None,
                "area": 0.0,
                "score": 0.0,
                "candidate_box": None,
                "area_ratio": 0.0,
                "aspect_ratio": 0.0,
                "rectangularity": 0.0,
            }

    draw_marker_overlay(output, detection, contour_state, status, preprocessing_mode)
    draw_frame_info(output, frame_index, total_frames, fps, paused)

    best_points = detection["corners"] if detection["found"] and detection.get("corners") is not None else None
    if show_contour_debug:
        debug_panel = prepare_contour_debug_panel(frame, contour_result)
    elif show_rejected_debug:
        debug_panel = prepare_debug_panel(frame, debug_images, rejected, best_points)
    else:
        debug_panel = None
    return output, debug_panel


def create_writer(video_path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter | None:
    if not SAVE_OUTPUT:
        return None
    output_path = video_path.with_name(f"{video_path.stem}_detected{video_path.suffix}")
    fourcc = cv2.VideoWriter_fourcc(*("mp4v" if output_path.suffix.lower() in {".mp4", ".m4v", ".mov"} else "XVID"))
    writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else DEFAULT_FPS, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Не удалось создать выходное видео: {output_path}")
    print(f"Запись результата: {output_path}")
    return writer


def print_controls(video_path: Path, dictionary_name: str, fps: float, total_frames: int) -> None:
    duration = format_time(total_frames / fps) if total_frames > 0 and fps > 0 else "неизвестна"
    print(f"Видео: {video_path}")
    print(f"Словарь: {dictionary_name}")
    print(f"Целевой маркер: ID {TARGET_MARKER_ID}")
    print(f"FPS: {fps:.2f}, кадров: {total_frames if total_frames > 0 else 'неизвестно'}, длительность: {duration}")
    print("Клавиши:")
    print("  Space - пауза / продолжение")
    print("  A/D   - назад / вперед на 1 секунду")
    print("  J/L   - назад / вперед на 5 секунд")
    print("  R     - перейти в начало")
    print("  M     - rejected candidates")
    print("  C     - диагностика контуров")
    print("  P     - режим предобработки")
    print("  Q/Esc - выход")


def main() -> int:
    global TARGET_MARKER_TEMPLATES

    args = parse_args()
    check_aruco_module()

    video_path = choose_video(args.video)
    if video_path is None:
        print("Видео не выбрано.")
        return 1

    TARGET_MARKER_TEMPLATES = create_target_marker_templates(args.dict, TARGET_ARUCO_ID, WARP_SIZE)
    detector = create_detector(args.dict)
    clahe = GLOBAL_CLAHE
    cap, fps, total_frames = open_video(video_path)
    contour_state = ContourTrackingState()
    writer: cv2.VideoWriter | None = None

    paused = False
    show_rejected_debug = False
    show_contour_debug = False
    preprocessing_index = 0
    seek_to: int | None = None
    current_frame: np.ndarray | None = None
    current_output: np.ndarray | None = None
    current_debug_panel: np.ndarray | None = None
    trackbar_update = False

    def on_trackbar(value: int) -> None:
        nonlocal seek_to
        if not trackbar_update:
            seek_to = value

    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.createTrackbar(TRACKBAR, WINDOW, 0, max(1, total_frames - 1), on_trackbar)
        print_controls(video_path, args.dict, fps, total_frames)

        while True:
            start_time = time.time()

            if seek_to is not None:
                frame_number = max(0, seek_to)
                if total_frames > 0:
                    frame_number = min(frame_number, total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ok, current_frame = cap.read()
                current_output = None
                current_debug_panel = None
                seek_to = None
                paused = True
                contour_state.reset()
                if not ok:
                    continue
            elif current_frame is None or not paused:
                ok, current_frame = cap.read()
                current_output = None
                current_debug_panel = None
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
                current_output, current_debug_panel = process_frame(
                    current_frame,
                    detector,
                    clahe,
                    contour_state,
                    frame_index,
                    total_frames,
                    fps,
                    paused,
                    show_rejected_debug,
                    show_contour_debug,
                    PREPROCESS_MODES[preprocessing_index],
                )
                if writer is None and SAVE_OUTPUT:
                    writer = create_writer(video_path, fps, (current_output.shape[1], current_output.shape[0]))
                if writer is not None and not paused:
                    writer.write(current_output)

            if current_output is not None:
                cv2.imshow(
                    WINDOW,
                    compose_display(current_output, current_debug_panel if (show_rejected_debug or show_contour_debug) else None),
                )

            if total_frames > 0:
                trackbar_update = True
                cv2.setTrackbarPos(TRACKBAR, WINDOW, min(frame_index, total_frames - 1))
                trackbar_update = False

            elapsed_ms = (time.time() - start_time) * 1000.0
            delay = 30 if paused else max(1, int(1000.0 / fps - elapsed_ms))
            key = cv2.waitKey(delay) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                current_output = None
            elif key in (ord("m"), ord("M")):
                show_rejected_debug = not show_rejected_debug
                if show_rejected_debug:
                    show_contour_debug = False
                current_output = None
                current_debug_panel = None
            elif key in (ord("c"), ord("C")):
                show_contour_debug = not show_contour_debug
                if show_contour_debug:
                    show_rejected_debug = False
                current_output = None
                current_debug_panel = None
            elif key in (ord("p"), ord("P")):
                preprocessing_index = (preprocessing_index + 1) % len(PREPROCESS_MODES)
                current_output = None
                current_debug_panel = None
            elif key in (ord("r"), ord("R")):
                seek_to = 0
                contour_state.reset()
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
                contour_state.reset()

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
