import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Keep Ultralytics settings inside the project when the user profile is not writable.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "Ultralytics"))

from ultralytics import YOLO


MODEL_PATH = "weights/best.pt"
VIDEO_PATH = "recordings/7.mp4"
OUTPUT_VIDEO_PATH = "output_yolo_aruco.mp4"

CONF_THRESHOLD = 0.7
IOU_THRESHOLD = 0.45
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20
MIN_BOX_AREA_RATIO = 0.001

ARUCO_DICTIONARY = "DICT_4X4_50"
ALLOWED_ARUCO_IDS = set()
ARUCO_ROI_MARGIN = 0.20
ARUCO_UPSCALE_FACTOR = 3.0
ARUCO_DUPLICATE_CENTER_DISTANCE = 20
ARUCO_CONFIRMATION_WINDOW = 0
ARUCO_RELATIVE_POSITION_TOLERANCE = 0.35

MIN_ARUCO_CONFIRMATIONS = 2
MIN_CLASS_MATCHES = 0
MIN_DRAW_OCCURRENCES = 0
CLASS_CHANGE_MARGIN = 0.15

CLASS_NAMES = {
    0: "Зарегистрированное",
    1: "Незарегистрированное",
}

DISPLAY_CLASS_NAMES = {
    0: "registered",
    1: "unregistered",
}

CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
}

PENDING_COLOR = (255, 255, 255)
ARUCO_COLOR = (255, 0, 255)
DEBUG_ROI_COLOR = (255, 0, 0)
TEXT_BG_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)

WINDOW_NAME = "YOLO + Robust ArUco"
DEFAULT_FPS = 30.0
DEBUG_DIR = "debug_aruco"


def parse_allowed_aruco_ids(value):
    if value is None:
        return set(ALLOWED_ARUCO_IDS)

    text = str(value).strip()
    if not text:
        return set(ALLOWED_ARUCO_IDS)

    allowed_ids = set()
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            marker_id = int(item)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                f"Invalid ArUco ID in --allowed-aruco-ids: {item}"
            ) from error
        if marker_id < 0:
            raise argparse.ArgumentTypeError(
                f"ArUco ID cannot be negative: {marker_id}"
            )
        allowed_ids.add(marker_id)

    return allowed_ids


def parse_args():
    parser = argparse.ArgumentParser(
        description="Обнаружение судов YOLOv8 и устойчивое сопоставление ArUco ID."
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Путь к модели YOLO .pt")
    parser.add_argument("--video", default=VIDEO_PATH, help="Путь к входному видео")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Порог confidence YOLO")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Порог IoU YOLO")
    parser.add_argument("--dictionary", default=ARUCO_DICTIONARY, help="Словарь ArUco")
    parser.add_argument("--min-box-width", type=int, default=MIN_BOX_WIDTH)
    parser.add_argument("--min-box-height", type=int, default=MIN_BOX_HEIGHT)
    parser.add_argument("--min-box-area-ratio", type=float, default=MIN_BOX_AREA_RATIO)
    parser.add_argument("--no-display", action="store_true", help="Не показывать окно OpenCV")
    parser.add_argument("--max-frames", type=int, default=0, help="0 означает обработать всё видео")
    parser.add_argument("--output", default=OUTPUT_VIDEO_PATH, help="Путь к выходному видео")
    parser.add_argument("--debug-aruco", action="store_true", help="Сохранять ROI и рисовать области поиска ArUco")
    parser.add_argument(
        "--allowed-aruco-ids",
        default="",
        help="Comma-separated ArUco whitelist, for example 0,1,2,3. Empty means any ID.",
    )
    args = parser.parse_args()
    args.allowed_aruco_ids = parse_allowed_aruco_ids(args.allowed_aruco_ids)
    return args


def load_model(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Файл модели не найден: {path}")

    print(f"Загрузка модели: {path}")
    model = YOLO(str(path))
    print(f"Классы в модели: {getattr(model, 'names', None)}")
    return model


def _set_aruco_parameter(parameters, name, value):
    if hasattr(parameters, name):
        setattr(parameters, name, value)


def create_aruco_detector(dictionary_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("cv2.aruco недоступен. Установите opencv-contrib-python.")

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Неизвестный словарь ArUco: {dictionary_name}")

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    strict_parameters = {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 31,
        "adaptiveThreshWinSizeStep": 4,
        "adaptiveThreshConstant": 7,
        "minMarkerPerimeterRate": 0.03,
        "maxMarkerPerimeterRate": 1.5,
        "polygonalApproxAccuracyRate": 0.04,
        "minCornerDistanceRate": 0.05,
        "minDistanceToBorder": 3,
        "perspectiveRemovePixelPerCell": 8,
        "perspectiveRemoveIgnoredMarginPerCell": 0.13,
        "errorCorrectionRate": 0.35,
        "maxErroneousBitsInBorderRate": 0.25,
        "useAruco3Detection": False,
    }

    for name, value in strict_parameters.items():
        _set_aruco_parameter(parameters, name, value)

    if hasattr(parameters, "cornerRefinementMethod"):
        if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters


def open_video(video_path):
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(f"Видео не найдено: {path}")

    print(f"Открытие видео: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Не удалось получить корректный размер видео.")
    if fps <= 0:
        fps = DEFAULT_FPS

    print(f"Размер видео: {width}x{height}")
    print(f"FPS видео: {fps:.2f}")
    return cap, width, height, fps


def create_video_writer(output_path, frame_width, frame_height, fps):
    if not output_path:
        print("Выходной файл не задан, запись видео отключена.")
        return None, None

    path = Path(output_path)
    if path.parent and str(path.parent) != ".":
        path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    if writer.isOpened():
        print(f"Запись видео: {path}")
        return writer, path

    writer.release()
    print("Предупреждение: mp4v writer не открылся, пробую AVI/MJPG.")
    fallback_path = path.with_suffix(".avi")
    writer = cv2.VideoWriter(
        str(fallback_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (frame_width, frame_height),
    )
    if writer.isOpened():
        print(f"Запись видео: {fallback_path}")
        return writer, fallback_path

    writer.release()
    print("Предупреждение: VideoWriter не открылся, обработка продолжится без записи.")
    return None, None


def clamp_box(x1, y1, x2, y2, frame_width, frame_height):
    x1 = max(0, min(int(round(x1)), frame_width - 1))
    y1 = max(0, min(int(round(y1)), frame_height - 1))
    x2 = max(0, min(int(round(x2)), frame_width - 1))
    y2 = max(0, min(int(round(y2)), frame_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_box(box, frame_width, frame_height, margin):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    dx = box_width * margin
    dy = box_height * margin
    return clamp_box(x1 - dx, y1 - dy, x2 + dx, y2 + dy, frame_width, frame_height)


def box_is_large_enough(box, frame_width, frame_height, args):
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    min_area = frame_width * frame_height * args.min_box_area_ratio

    return (
        box_width >= args.min_box_width
        and box_height >= args.min_box_height
        and box_area >= min_area
    )


def get_yolo_detections(model, frame, args):
    frame_height, frame_width = frame.shape[:2]
    result = model.predict(
        source=frame,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )[0]

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    detections = []
    for detection_index, box in enumerate(boxes):
        xyxy = box.xyxy[0].detach().cpu().tolist()
        class_id = int(box.cls[0].detach().cpu().item())
        confidence = float(box.conf[0].detach().cpu().item())

        clamped = clamp_box(*xyxy, frame_width, frame_height)
        if clamped is None or not box_is_large_enough(clamped, frame_width, frame_height, args):
            continue

        expanded = expand_box(clamped, frame_width, frame_height, ARUCO_ROI_MARGIN)
        detections.append(
            {
                "index": detection_index,
                "box": clamped,
                "expanded_box": expanded if expanded is not None else clamped,
                "class_id": class_id,
                "confidence": confidence,
            }
        )

    return detections


def create_aruco_image_variants(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    clahe = cv2.createCLAHE(
        clipLimit=1.5,
        tileGridSize=(8, 8),
    )
    gray_clahe = clahe.apply(gray)

    return [
        ("gray", gray),
        ("clahe", gray_clahe),
    ]


def detect_aruco_on_image(image, detector):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=parameters,
        )

    rejected_count = 0 if rejected is None else len(rejected)
    if ids is None or len(ids) == 0:
        return [], rejected_count

    markers = []
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        corners_array = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        markers.append(
            {
                "id": int(marker_id),
                "corners": corners_array,
                "center": get_marker_center(corners_array),
                "area": get_marker_area(corners_array),
                "source": "image",
            }
        )

    return markers, rejected_count


def _should_save_debug_roi(debug_last_save_times, detection_index):
    now = time.time()
    last_time = debug_last_save_times.get(detection_index, 0.0)
    if now - last_time < 1.0:
        return False
    debug_last_save_times[detection_index] = now
    return True


def _record_discarded_aruco(discarded_aruco_ids, marker_id, reason):
    if discarded_aruco_ids is None:
        return
    entry = discarded_aruco_ids.setdefault(
        int(marker_id),
        {
            "count": 0,
            "reasons": {},
        },
    )
    entry["count"] += 1
    entry["reasons"][reason] = entry["reasons"].get(reason, 0) + 1


def _save_debug_roi_images(roi, variants, frame_number, detection_index):
    debug_path = Path(DEBUG_DIR)
    debug_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(
        str(debug_path / f"frame_{frame_number:06d}_det_{detection_index:02d}_original.jpg"),
        roi,
    )
    for variant_name, variant in variants:
        cv2.imwrite(
            str(debug_path / f"frame_{frame_number:06d}_det_{detection_index:02d}_{variant_name}.jpg"),
            variant,
        )


def _print_debug_aruco_candidate(
    frame_number,
    detection_index,
    marker_id,
    source,
    geometry_info,
    geometry_ok,
    whitelist_ok,
    relative_center=None,
    recent_confirmations=0,
):
    sides_text = ",".join(f"{side:.1f}" for side in geometry_info["side_lengths"])
    area_ratio = geometry_info["area_ratio"]
    relative_text = "n/a"
    if relative_center is not None:
        relative_text = f"({relative_center[0]:.2f},{relative_center[1]:.2f})"
    print(
        f"DEBUG ArUco frame={frame_number} det={detection_index} "
        f"id={marker_id} source={source} area={geometry_info['area']:.1f} "
        f"sides=[{sides_text}] area_ratio={area_ratio:.5f} "
        f"geometry_ok={geometry_ok} whitelist_ok={whitelist_ok} "
        f"relative_center={relative_text} recent={recent_confirmations}/{MIN_ARUCO_CONFIRMATIONS}"
    )


def detect_aruco_in_roi(
    frame,
    detection,
    detector,
    frame_number,
    allowed_ids,
    debug_aruco=False,
    debug_last_save_times=None,
    discarded_aruco_ids=None,
):
    box = detection["expanded_box"]
    detection_index = detection["index"]
    x1, y1, x2, y2 = box
    try:
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return [], 0

        upscaled_roi = cv2.resize(
            roi,
            None,
            fx=ARUCO_UPSCALE_FACTOR,
            fy=ARUCO_UPSCALE_FACTOR,
            interpolation=cv2.INTER_CUBIC,
        )

        variants = create_aruco_image_variants(upscaled_roi)
        if (
            debug_aruco
            and debug_last_save_times is not None
            and _should_save_debug_roi(debug_last_save_times, detection_index)
        ):
            _save_debug_roi_images(roi, variants, frame_number, detection_index)

        all_markers = []
        total_rejected = 0
        roi_height, roi_width = roi.shape[:2]
        for variant_name, variant in variants:
            markers, rejected_count = detect_aruco_on_image(variant, detector)
            total_rejected += rejected_count
            for marker in markers:
                marker_id = int(marker["id"])
                corners = marker["corners"].astype(np.float32)
                corners /= ARUCO_UPSCALE_FACTOR
                geometry_info = get_marker_geometry_info(corners, roi_width, roi_height)
                geometry_ok = marker_geometry_is_valid(corners, roi_width, roi_height)
                whitelist_ok = not allowed_ids or marker_id in allowed_ids

                if debug_aruco:
                    _print_debug_aruco_candidate(
                        frame_number,
                        detection_index,
                        marker_id,
                        variant_name,
                        geometry_info,
                        geometry_ok,
                        whitelist_ok,
                    )

                if not geometry_ok:
                    _record_discarded_aruco(discarded_aruco_ids, marker_id, "geometry")
                    continue
                if not whitelist_ok:
                    _record_discarded_aruco(discarded_aruco_ids, marker_id, "whitelist")
                    continue

                corners[:, 0] += x1
                corners[:, 1] += y1
                marker["corners"] = corners
                marker["center"] = get_marker_center(corners)
                marker["area"] = get_marker_area(corners)
                marker["source"] = variant_name
                marker["geometry"] = geometry_info
                all_markers.append(marker)

        return deduplicate_markers(all_markers), total_rejected

    except Exception as error:
        print(
            f"Предупреждение: ошибка ArUco ROI на кадре {frame_number}, "
            f"детекция {detection_index}: {type(error).__name__}: {error}"
        )
        return [], 0


def get_marker_center(corners):
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    center_x = float(np.mean(corners[:, 0]))
    center_y = float(np.mean(corners[:, 1]))
    return center_x, center_y


def get_marker_area(corners):
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return float(abs(cv2.contourArea(corners)))


def get_marker_side_lengths(corners):
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return [
        float(np.linalg.norm(corners[(index + 1) % 4] - corners[index]))
        for index in range(4)
    ]


def get_marker_diagonal_lengths(corners):
    corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    return [
        float(np.linalg.norm(corners[2] - corners[0])),
        float(np.linalg.norm(corners[3] - corners[1])),
    ]


def get_marker_geometry_info(corners, roi_width, roi_height):
    try:
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    except ValueError:
        return {
            "shape_ok": False,
            "finite": False,
            "side_lengths": [],
            "diagonal_lengths": [],
            "area": 0.0,
            "area_ratio": 0.0,
            "convex": False,
        }

    area = get_marker_area(corners) if np.all(np.isfinite(corners)) else 0.0
    roi_area = max(1.0, float(roi_width * roi_height))
    return {
        "shape_ok": corners.shape == (4, 2),
        "finite": bool(np.all(np.isfinite(corners))),
        "side_lengths": get_marker_side_lengths(corners),
        "diagonal_lengths": get_marker_diagonal_lengths(corners),
        "area": area,
        "area_ratio": area / roi_area,
        "convex": bool(cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2))),
    }


def marker_geometry_is_valid(corners, roi_width, roi_height):
    try:
        corners = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    except ValueError:
        return False

    if corners.shape != (4, 2):
        return False
    if not np.all(np.isfinite(corners)):
        return False

    side_lengths = get_marker_side_lengths(corners)
    min_side = min(side_lengths)
    max_side = max(side_lengths)
    if min_side < 8.0:
        return False
    if max_side / max(min_side, 1e-6) > 2.0:
        return False

    area = get_marker_area(corners)
    if area <= 0.0:
        return False

    area_ratio = area / max(1.0, float(roi_width * roi_height))
    if area_ratio < 0.001 or area_ratio > 0.30:
        return False

    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        return False

    diagonal_lengths = get_marker_diagonal_lengths(corners)
    min_diagonal = min(diagonal_lengths)
    max_diagonal = max(diagonal_lengths)
    if max_diagonal / max(min_diagonal, 1e-6) > 2.0:
        return False

    return True


def _marker_bbox(marker):
    corners = marker["corners"]
    x1 = float(np.min(corners[:, 0]))
    y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0]))
    y2 = float(np.max(corners[:, 1]))
    return x1, y1, x2, y2


def _bbox_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    intersection_width = max(0.0, ix2 - ix1)
    intersection_height = max(0.0, iy2 - iy1)
    intersection = intersection_width * intersection_height
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _markers_are_duplicates(marker_a, marker_b):
    if marker_a["id"] != marker_b["id"]:
        return False

    ax, ay = marker_a["center"]
    bx, by = marker_b["center"]
    center_distance = float(np.hypot(ax - bx, ay - by))
    if center_distance < ARUCO_DUPLICATE_CENTER_DISTANCE:
        return True

    return _bbox_iou(_marker_bbox(marker_a), _marker_bbox(marker_b)) >= 0.35


def deduplicate_markers(markers):
    unique_markers = []
    for marker in sorted(markers, key=lambda item: item.get("area", 0.0), reverse=True):
        duplicate_index = None
        for index, existing_marker in enumerate(unique_markers):
            if _markers_are_duplicates(marker, existing_marker):
                duplicate_index = index
                break

        if duplicate_index is None:
            unique_markers.append(marker)
            continue

        if marker.get("area", 0.0) > unique_markers[duplicate_index].get("area", 0.0):
            unique_markers[duplicate_index] = marker

    return unique_markers


def _point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _box_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def get_normalized_marker_center(marker, detection):
    marker_center_x, marker_center_y = marker["center"]
    x1, y1, x2, y2 = detection["box"]
    box_width = max(1.0, float(x2 - x1))
    box_height = max(1.0, float(y2 - y1))
    relative_x = (marker_center_x - x1) / box_width
    relative_y = (marker_center_y - y1) / box_height
    return float(relative_x), float(relative_y)


def normalized_center_is_reasonable(center):
    relative_x, relative_y = center
    return (
        -0.20 <= relative_x <= 1.20
        and -0.20 <= relative_y <= 1.20
    )


def normalized_centers_are_consistent(centers):
    if len(centers) < MIN_ARUCO_CONFIRMATIONS:
        return False

    xs = [center[0] for center in centers]
    ys = [center[1] for center in centers]
    return (
        max(xs) - min(xs) <= ARUCO_RELATIVE_POSITION_TOLERANCE
        and max(ys) - min(ys) <= ARUCO_RELATIVE_POSITION_TOLERANCE
    )


def match_markers_to_detections(markers, detections, frame_number=None):
    candidate_by_marker = []
    used_marker_ids = set()

    for marker in sorted(markers, key=lambda item: item.get("area", 0.0), reverse=True):
        marker_id = marker["id"]
        if marker_id in used_marker_ids:
            continue

        marker_center = marker["center"]
        candidates = []
        for detection in detections:
            normal_match = _point_in_box(marker_center, detection["box"])
            expanded_match = _point_in_box(marker_center, detection["expanded_box"])
            if not normal_match and not expanded_match:
                continue

            relative_center = get_normalized_marker_center(marker, detection)
            if not normalized_center_is_reasonable(relative_center):
                continue

            detection_center = _box_center(detection["box"])
            distance = float(
                np.hypot(
                    marker_center[0] - detection_center[0],
                    marker_center[1] - detection_center[1],
                )
            )
            candidates.append((distance, -float(detection["confidence"]), detection, relative_center))

        if not candidates:
            continue

        candidates.sort(key=lambda item: (item[0], item[1]))
        candidate_by_marker.append(
            {
                "marker": marker,
                "detection": candidates[0][2],
                "distance": candidates[0][0],
                "relative_center": candidates[0][3],
            }
        )
        used_marker_ids.add(marker_id)

    markers_by_detection = {}
    for item in candidate_by_marker:
        detection_index = item["detection"]["index"]
        markers_by_detection.setdefault(detection_index, []).append(item)

    matches_by_detection = {}
    matches_by_id = {}
    unused_markers = []

    for detection_index, items in markers_by_detection.items():
        items.sort(key=lambda item: item["marker"].get("area", 0.0), reverse=True)
        best_item = items[0]
        best_marker = best_item["marker"]
        matches_by_detection[detection_index] = best_item
        matches_by_id[best_marker["id"]] = best_item

        other_ids = sorted(
            {
                item["marker"]["id"]
                for item in items[1:]
                if item["marker"]["id"] != best_marker["id"]
            }
        )
        if other_ids:
            unused_markers.extend(item["marker"] for item in items[1:])
            frame_text = f" на кадре {frame_number}" if frame_number is not None else ""
            print(
                f"Предупреждение: в одной YOLO-рамке{frame_text} найдено несколько ArUco ID: "
                f"использую ID {best_marker['id']}, остальные {other_ids} только отрисованы."
            )

    return {
        "by_detection": matches_by_detection,
        "by_id": matches_by_id,
        "unused_markers": unused_markers,
    }


def _create_vessel_entry(aruco_id, frame_number):
    return {
        "aruco_id": aruco_id,
        "occurrences": 0,
        "matches": 0,
        "registered_score": 0.0,
        "unregistered_score": 0.0,
        "registered_frames": 0,
        "unregistered_frames": 0,
        "first_seen_frame": frame_number,
        "last_seen_frame": frame_number,
        "last_box": None,
        "best_confidence": 0.0,
        "final_class_id": None,
        "confirmed": False,
        "reported": False,
        "recent_detection_frames": [],
        "recent_centers_normalized": [],
        "recent_center_frames": [],
        "candidate_confirmed": False,
    }


def _score_for_class(vessel, class_id):
    if class_id == 0:
        return vessel["registered_score"]
    if class_id == 1:
        return vessel["unregistered_score"]
    return 0.0


def resolve_vessel_class(vessel):
    previous_class_id = vessel["final_class_id"]
    registered_score = vessel["registered_score"]
    unregistered_score = vessel["unregistered_score"]

    if registered_score > unregistered_score:
        candidate_class_id = 0
    elif unregistered_score > registered_score:
        candidate_class_id = 1
    elif vessel["registered_frames"] > vessel["unregistered_frames"]:
        candidate_class_id = 0
    elif vessel["unregistered_frames"] > vessel["registered_frames"]:
        candidate_class_id = 1
    else:
        candidate_class_id = previous_class_id

    if candidate_class_id is None:
        return previous_class_id, False

    if previous_class_id is None:
        vessel["final_class_id"] = candidate_class_id
        return candidate_class_id, False

    if candidate_class_id == previous_class_id:
        return previous_class_id, False

    if vessel["confirmed"]:
        previous_score = _score_for_class(vessel, previous_class_id)
        candidate_score = _score_for_class(vessel, candidate_class_id)
        if candidate_score <= previous_score * (1.0 + CLASS_CHANGE_MARGIN):
            return previous_class_id, False

    vessel["final_class_id"] = candidate_class_id
    return candidate_class_id, True


def _trim_vessel_confirmation_window(vessel, frame_number):
    oldest_allowed_frame = frame_number - ARUCO_CONFIRMATION_WINDOW
    vessel["recent_detection_frames"] = [
        frame
        for frame in vessel["recent_detection_frames"]
        if frame >= oldest_allowed_frame
    ]

    filtered_centers = []
    filtered_center_frames = []
    for center, center_frame in zip(
        vessel["recent_centers_normalized"],
        vessel["recent_center_frames"],
    ):
        if center_frame >= oldest_allowed_frame:
            filtered_centers.append(center)
            filtered_center_frames.append(center_frame)

    vessel["recent_centers_normalized"] = filtered_centers
    vessel["recent_center_frames"] = filtered_center_frames


def _update_vessel_confirmation_state(vessel, allowed_ids):
    whitelist_ok = not allowed_ids or vessel["aruco_id"] in allowed_ids
    total_confirmations = vessel["occurrences"]
    vessel["candidate_confirmed"] = (
        total_confirmations >= MIN_ARUCO_CONFIRMATIONS
        and whitelist_ok
    )
    vessel["confirmed"] = (
        total_confirmations >= MIN_ARUCO_CONFIRMATIONS
        and vessel["matches"] >= MIN_CLASS_MATCHES
        and vessel["final_class_id"] is not None
        and whitelist_ok
    )


def update_vessel_registry(
    vessels,
    markers,
    matches,
    frame_number,
    allowed_ids,
    debug_aruco=False,
):
    class_change_events = []
    seen_ids_this_frame = set()

    for marker in markers:
        aruco_id = marker["id"]
        if aruco_id in seen_ids_this_frame:
            continue
        seen_ids_this_frame.add(aruco_id)

        if aruco_id not in vessels:
            vessels[aruco_id] = _create_vessel_entry(aruco_id, frame_number)

        vessel = vessels[aruco_id]
        vessel["occurrences"] += 1
        vessel["last_seen_frame"] = frame_number
        vessel["recent_detection_frames"].append(frame_number)
        _trim_vessel_confirmation_window(vessel, frame_number)
        _update_vessel_confirmation_state(vessel, allowed_ids)

    matched_ids_this_frame = set()

    for aruco_id, match in matches["by_id"].items():
        if aruco_id in matched_ids_this_frame:
            continue
        matched_ids_this_frame.add(aruco_id)

        if aruco_id not in vessels:
            vessels[aruco_id] = _create_vessel_entry(aruco_id, frame_number)

        detection = match["detection"]
        vessel = vessels[aruco_id]
        class_id = int(detection["class_id"])
        confidence = float(detection["confidence"])
        relative_center = match["relative_center"]

        if not normalized_center_is_reasonable(relative_center):
            continue

        vessel["matches"] += 1
        vessel["last_seen_frame"] = frame_number
        vessel["last_box"] = detection["box"]
        vessel["best_confidence"] = max(vessel["best_confidence"], confidence)
        vessel["recent_centers_normalized"].append(relative_center)
        vessel["recent_center_frames"].append(frame_number)
        _trim_vessel_confirmation_window(vessel, frame_number)

        if class_id == 0:
            vessel["registered_score"] += confidence
            vessel["registered_frames"] += 1
        elif class_id == 1:
            vessel["unregistered_score"] += confidence
            vessel["unregistered_frames"] += 1

        was_confirmed = vessel["confirmed"]
        old_class_id = vessel["final_class_id"]
        new_class_id, class_changed = resolve_vessel_class(vessel)
        _update_vessel_confirmation_state(vessel, allowed_ids)

        if debug_aruco:
            _print_debug_aruco_candidate(
                frame_number,
                detection["index"],
                aruco_id,
                match["marker"].get("source", "roi"),
                match["marker"].get(
                    "geometry",
                    get_marker_geometry_info(match["marker"]["corners"], 1, 1),
                ),
                True,
                not allowed_ids or aruco_id in allowed_ids,
                relative_center=relative_center,
                recent_confirmations=len(vessel["recent_detection_frames"]),
            )

        if (
            class_changed
            and old_class_id is not None
            and new_class_id is not None
            and was_confirmed
            and vessel["confirmed"]
        ):
            class_change_events.append((aruco_id, new_class_id))

    return class_change_events


def get_vessel_statistics(vessels):
    registered = 0
    unregistered = 0
    pending = 0

    for vessel in vessels.values():
        if vessel["confirmed"] and vessel["final_class_id"] == 0:
            registered += 1
        elif vessel["confirmed"] and vessel["final_class_id"] == 1:
            unregistered += 1
        else:
            pending += 1

    return {
        "unique": registered + unregistered,
        "registered": registered,
        "unregistered": unregistered,
        "pending": pending,
    }


def _get_text_color(background_color):
    blue, green, red = background_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness > 140 else TEXT_COLOR


def draw_text(frame, text, position, background_color=TEXT_BG_COLOR, scale=0.55, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, scale, thickness)
    text_width, text_height = text_size
    frame_height, frame_width = frame.shape[:2]

    x = max(0, min(int(position[0]), max(0, frame_width - text_width - 8)))
    y = max(text_height + 8, min(int(position[1]), frame_height - baseline - 4))

    cv2.rectangle(
        frame,
        (x - 4, y - text_height - baseline - 4),
        (x + text_width + 4, y + baseline + 4),
        background_color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        _get_text_color(background_color),
        thickness,
        cv2.LINE_AA,
    )


def draw_yolo_detection(frame, detection, matches_by_detection, vessels, debug_aruco=False):
    x1, y1, x2, y2 = detection["box"]
    class_id = detection["class_id"]
    confidence = detection["confidence"]
    color = CLASS_COLORS.get(class_id, PENDING_COLOR)
    detection_index = detection["index"]
    match = matches_by_detection.get(detection_index)

    if debug_aruco:
        ex1, ey1, ex2, ey2 = detection["expanded_box"]
        cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), DEBUG_ROI_COLOR, 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if match is None:
        class_name = DISPLAY_CLASS_NAMES.get(class_id, f"class_{class_id}")
        label = f"ID:unknown | {class_name} | {confidence:.2f}"
        draw_text(frame, label, (x1, y1 - 8), color)
        return

    aruco_id = match["marker"]["id"]
    vessel = vessels.get(aruco_id)
    if (
        not debug_aruco
        and (vessel is None or vessel["occurrences"] < MIN_DRAW_OCCURRENCES)
    ):
        class_name = DISPLAY_CLASS_NAMES.get(class_id, f"class_{class_id}")
        label = f"ID:unknown | {class_name} | {confidence:.2f}"
        draw_text(frame, label, (x1, y1 - 8), color)
        return

    if vessel is not None and vessel["confirmed"] and vessel["final_class_id"] is not None:
        final_class_id = int(vessel["final_class_id"])
        class_name = DISPLAY_CLASS_NAMES.get(final_class_id, f"class_{final_class_id}")
        label = f"ID:{aruco_id} | {class_name} | {confidence:.2f}"
    else:
        total_confirmations = 0 if vessel is None else vessel["occurrences"]
        label = f"ID:{aruco_id} | candidate {total_confirmations}/{MIN_ARUCO_CONFIRMATIONS}"

    draw_text(frame, label, (x1, y1 - 8), color)


def draw_aruco_marker(frame, marker):
    corners = marker["corners"].astype(np.int32)
    marker_id = marker["id"]
    center = tuple(np.mean(corners, axis=0).astype(int))

    cv2.polylines(frame, [corners], True, ARUCO_COLOR, 2)
    cv2.circle(frame, center, 4, ARUCO_COLOR, -1)
    draw_text(frame, f"ID:{marker_id}", (center[0] + 6, center[1] - 6), ARUCO_COLOR, scale=0.5)


def marker_should_be_drawn(marker, vessels, debug_aruco=False):
    if debug_aruco:
        return True
    vessel = vessels.get(marker["id"])
    return vessel is not None and (
        vessel["confirmed"] or vessel["occurrences"] >= MIN_DRAW_OCCURRENCES
    )


def draw_statistics(frame, fps, detections_count, markers_count, vessels, rejected_count=0, debug_aruco=False):
    stats = get_vessel_statistics(vessels)
    lines = [
        f"FPS: {fps:.1f}",
        f"YOLO objects: {detections_count}",
        f"ArUco markers: {markers_count}",
        f"Unique vessels: {stats['unique']}",
        f"Registered: {stats['registered']}",
        f"Unregistered: {stats['unregistered']}",
        f"Pending: {stats['pending']}",
    ]
    if debug_aruco:
        lines.append(f"Rejected: {rejected_count}")

    y = 28
    for line in lines:
        draw_text(frame, line, (12, y), TEXT_BG_COLOR, scale=0.55)
        y += 26


def print_new_vessel_events(vessels, class_change_events):
    for aruco_id, vessel in sorted(vessels.items()):
        if (
            vessel["confirmed"]
            and vessel["final_class_id"] is not None
            and not vessel["reported"]
        ):
            class_name = CLASS_NAMES.get(vessel["final_class_id"], f"Класс {vessel['final_class_id']}").lower()
            print(
                f"Обнаружено: {class_name} судно, ArUco ID: {aruco_id}, "
                f"confidence: {vessel['best_confidence']:.2f}, совпадений: {vessel['matches']}"
            )
            print(f"Кадров с прочитанным ID: {vessel['occurrences']}")
            vessel["reported"] = True

    for aruco_id, new_class_id in class_change_events:
        vessel = vessels.get(aruco_id)
        if vessel is None or not vessel["confirmed"]:
            continue
        class_name = CLASS_NAMES.get(new_class_id, f"Класс {new_class_id}").lower()
        print(f"Обновлён класс: ArUco ID {aruco_id} — {class_name} судно")


def print_final_report(vessels):
    confirmed = [
        vessel
        for vessel in vessels.values()
        if vessel["confirmed"] and vessel["final_class_id"] is not None
    ]
    registered = sum(1 for vessel in confirmed if vessel["final_class_id"] == 0)
    unregistered = sum(1 for vessel in confirmed if vessel["final_class_id"] == 1)
    pending = [
        vessel
        for vessel in vessels.values()
        if not (vessel["confirmed"] and vessel["final_class_id"] is not None)
    ]

    print()
    print("==================================================")


    print("ИТОГОВЫЙ ОТЧЁТ")
    print("==================================================")
    print(f"Всего уникальных подтверждённых судов: {len(confirmed)}")
    print(f"Зарегистрированных: {registered}")
    print(f"Незарегистрированных: {unregistered}")
    print(f"Неподтверждённых ID: {len(pending)}")
    print()
    print("Список подтверждённых судов:")
    if confirmed:
        for vessel in sorted(confirmed, key=lambda item: item["aruco_id"]):
            class_name = CLASS_NAMES.get(vessel["final_class_id"], f"Класс {vessel['final_class_id']}").lower()
            print(f"ArUco ID {vessel['aruco_id']} — {class_name}")
    else:
        print("Нет")

    print()
    print("Неподтверждённые:")
    if pending:
        for vessel in sorted(pending, key=lambda item: item["aruco_id"]):
            print(
                f"ArUco ID {vessel['aruco_id']} — "
                f"occurrences: {vessel['occurrences']}, matches: {vessel['matches']}"
            )
    else:
        print("Нет")
    print("==================================================")


def get_unconfirmed_reasons(vessel, allowed_ids):
    reasons = []
    recent_confirmations = len(vessel["recent_detection_frames"])

    if allowed_ids and vessel["aruco_id"] not in allowed_ids:
        reasons.append("ID запрещён белым списком")
    if vessel["occurrences"] < MIN_ARUCO_CONFIRMATIONS:
        reasons.append("недостаточно повторений")
    if vessel["matches"] < MIN_CLASS_MATCHES:
        reasons.append("недостаточно совпадений с YOLO")
    if (
        recent_confirmations >= MIN_ARUCO_CONFIRMATIONS
        and not normalized_centers_are_consistent(vessel["recent_centers_normalized"])
    ):
        reasons.append("нестабильное положение")
    if vessel["final_class_id"] is None:
        reasons.append("итоговый класс не определён")

    return reasons or ["не подтверждён"]


def print_final_report(
    vessels,
    allowed_ids=None,
    final_frame_number=None,
    debug_aruco=False,
    discarded_aruco_ids=None,
):
    allowed_ids = allowed_ids or set()
    if final_frame_number is not None:
        for vessel in vessels.values():
            _trim_vessel_confirmation_window(vessel, final_frame_number)
            _update_vessel_confirmation_state(vessel, allowed_ids)

    confirmed = [
        vessel
        for vessel in vessels.values()
        if vessel["confirmed"] and vessel["final_class_id"] is not None
    ]
    registered = sum(1 for vessel in confirmed if vessel["final_class_id"] == 0)
    unregistered = sum(1 for vessel in confirmed if vessel["final_class_id"] == 1)
    pending = [
        vessel
        for vessel in vessels.values()
        if not (vessel["confirmed"] and vessel["final_class_id"] is not None)
    ]

    print()
    print("==================================================")
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("==================================================")
    print(f"Всего уникальных подтверждённых судов: {len(confirmed)}")
    print(f"Зарегистрированных: {registered}")
    print(f"Незарегистрированных: {unregistered}")
    print(f"Кандидатов без подтверждения: {len(pending)}")
    print()
    print("Подтверждённые суда:")
    if confirmed:
        for vessel in sorted(confirmed, key=lambda item: item["aruco_id"]):
            class_name = CLASS_NAMES.get(vessel["final_class_id"], f"Класс {vessel['final_class_id']}").lower()
            print(f"ArUco ID {vessel['aruco_id']} — {class_name}")
    else:
        print("Нет")

    print()
    print("Кандидаты, которые не прошли подтверждение:")
    if pending:
        for vessel in sorted(pending, key=lambda item: item["aruco_id"]):
            reasons = "; ".join(get_unconfirmed_reasons(vessel, allowed_ids))
            print(
                f"ArUco ID {vessel['aruco_id']} — "
                f"occurrences: {vessel['occurrences']}, "
                f"occurrences в окне: {len(vessel['recent_detection_frames'])}, "
                f"matches: {vessel['matches']}, причина: {reasons}"
            )
    else:
        print("Нет")

    if debug_aruco and discarded_aruco_ids:
        print()
        print("Отброшенные ложные ID:")
        for marker_id in sorted(discarded_aruco_ids):
            entry = discarded_aruco_ids[marker_id]
            reasons = ", ".join(
                f"{reason}: {count}"
                for reason, count in sorted(entry["reasons"].items())
            )
            print(f"ArUco ID {marker_id} — count: {entry['count']}, reasons: {reasons}")

    print("==================================================")


def wait_delay_ms(source_fps, frame_start_time):
    target_time = 1.0 / source_fps if source_fps > 0 else 1.0 / DEFAULT_FPS
    elapsed = time.time() - frame_start_time
    remaining = target_time - elapsed
    if remaining <= 0:
        return 1
    return max(1, int(remaining * 1000))


def main():
    args = parse_args()
    cap = None
    writer = None
    processed_frames = 0
    total_time = 0.0
    vessels = {}
    debug_last_save_times = {}
    discarded_aruco_ids = {}

    try:
        yolo_model = load_model(args.model)
        aruco_detector = create_aruco_detector(args.dictionary)
        cap, width, height, source_fps = open_video(args.video)
        writer, writer_path = create_video_writer(args.output, width, height, source_fps)
        if args.allowed_aruco_ids:
            allowed_text = ", ".join(str(marker_id) for marker_id in sorted(args.allowed_aruco_ids))
            print(f"Белый список ArUco ID: {allowed_text}")
        else:
            print("Белый список ArUco ID: не задан, разрешены любые ID")

        print(f"ArUco словарь: {args.dictionary}")
        print(f"Расширение ROI ArUco: {ARUCO_ROI_MARGIN:.2f}")
        print(f"Увеличение ROI ArUco: {ARUCO_UPSCALE_FACTOR:.1f}x")
        print("Начало обработки.")

        while True:
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            frame_start_time = time.time()
            frame_number = processed_frames + 1

            detections = get_yolo_detections(yolo_model, frame, args)

            frame_markers = []
            total_rejected_count = 0

            for detection in detections:
                roi_markers, roi_rejected_count = detect_aruco_in_roi(
                    frame,
                    detection,
                    aruco_detector,
                    frame_number,
                    args.allowed_aruco_ids,
                    debug_aruco=args.debug_aruco,
                    debug_last_save_times=debug_last_save_times,
                    discarded_aruco_ids=discarded_aruco_ids,
                )
                frame_markers.extend(roi_markers)
                total_rejected_count += roi_rejected_count

            frame_markers = deduplicate_markers(frame_markers)
            matches = match_markers_to_detections(frame_markers, detections, frame_number)
            class_change_events = update_vessel_registry(
                vessels,
                frame_markers,
                matches,
                frame_number,
                args.allowed_aruco_ids,
                debug_aruco=args.debug_aruco,
            )
            print_new_vessel_events(vessels, class_change_events)

            visible_markers = [
                marker
                for marker in frame_markers
                if marker_should_be_drawn(marker, vessels, args.debug_aruco)
            ]

            for marker in visible_markers:
                draw_aruco_marker(frame, marker)

            for detection in detections:
                draw_yolo_detection(
                    frame,
                    detection,
                    matches["by_detection"],
                    vessels,
                    debug_aruco=args.debug_aruco,
                )

            frame_time = time.time() - frame_start_time
            total_time += frame_time
            processed_frames += 1
            processing_fps = 1.0 / frame_time if frame_time > 0 else 0.0

            draw_statistics(
                frame,
                processing_fps,
                len(detections),
                len(visible_markers),
                vessels,
                rejected_count=total_rejected_count,
                debug_aruco=args.debug_aruco,
            )

            if writer is not None:
                writer.write(frame)

            if processed_frames % 30 == 0:
                stats = get_vessel_statistics(vessels)
                print(
                    f"Кадр {processed_frames} | "
                    f"YOLO: {len(detections)} | "
                    f"ArUco: {len(visible_markers)} | "
                    f"подтверждённых судов: {stats['unique']}"
                )

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(wait_delay_ms(source_fps, frame_start_time)) & 0xFF
                if key in (ord("q"), 27):
                    break

        average_fps = processed_frames / total_time if total_time > 0 else 0.0
        print()
        print("Обработка завершена.")
        print(f"Обработано кадров: {processed_frames}")
        print(f"Средний FPS обработки: {average_fps:.1f}")
        if writer is not None:
            print(f"Обработанное видео сохранено: {writer_path}")

    except Exception as error:
        print(f"Ошибка: {type(error).__name__}: {error}", file=sys.stderr)
        raise

    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print_final_report(
            vessels,
            allowed_ids=getattr(args, "allowed_aruco_ids", set()),
            final_frame_number=processed_frames,
            debug_aruco=getattr(args, "debug_aruco", False),
            discarded_aruco_ids=discarded_aruco_ids,
        )


def get_terminal_class_name(class_id):
    if class_id == 0:
        return "зарегистрированное"
    if class_id == 1:
        return "незарегистрированное"
    return f"неизвестный класс {class_id}"


def print_final_report(
    vessels,
    allowed_ids=None,
    final_frame_number=None,
    debug_aruco=False,
    discarded_aruco_ids=None,
):
    allowed_ids = allowed_ids or set()
    if final_frame_number is not None:
        for vessel in vessels.values():
            _trim_vessel_confirmation_window(vessel, final_frame_number)
            _update_vessel_confirmation_state(vessel, allowed_ids)

    confirmed = sorted(
        (
            vessel
            for vessel in vessels.values()
            if vessel["confirmed"] and vessel["final_class_id"] is not None
        ),
        key=lambda item: item["aruco_id"],
    )
    candidates = sorted(
        (
            vessel
            for vessel in vessels.values()
            if not (vessel["confirmed"] and vessel["final_class_id"] is not None)
        ),
        key=lambda item: item["aruco_id"],
    )

    registered = sum(1 for vessel in confirmed if vessel["final_class_id"] == 0)
    unregistered = sum(1 for vessel in confirmed if vessel["final_class_id"] == 1)

    print()
    print("==================================================")
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("==================================================")
    print(f"Всего подтверждённых судов: {len(confirmed)}")
    print(f"Зарегистрированных: {registered}")
    print(f"Незарегистрированных: {unregistered}")
    print()
    print("Суда:")

    if confirmed:
        for number, vessel in enumerate(confirmed, start=1):
            class_name = get_terminal_class_name(vessel["final_class_id"])
            print(
                f"{number}. ArUco ID {vessel['aruco_id']} — "
                f"{class_name} судно"
            )
    else:
        print("Подтверждённые суда не найдены.")

    if candidates:
        print()
        print("Кандидаты без подтверждения:")
        for vessel in candidates:
            reasons = "; ".join(get_unconfirmed_reasons(vessel, allowed_ids))
            print(
                f"ArUco ID {vessel['aruco_id']} — "
                f"occurrences: {vessel['occurrences']}, "
                f"occurrences в окне: {len(vessel['recent_detection_frames'])}, "
                f"matches: {vessel['matches']}, причина: {reasons}"
            )

    if debug_aruco and discarded_aruco_ids:
        print()
        print("Отброшенные ID:")
        for marker_id in sorted(discarded_aruco_ids):
            entry = discarded_aruco_ids[marker_id]
            reasons = ", ".join(
                f"{reason}: {count}"
                for reason, count in sorted(entry["reasons"].items())
            )
            print(f"ArUco ID {marker_id} — count: {entry['count']}, reasons: {reasons}")

    print("==================================================")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)
