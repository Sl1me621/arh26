import time
from typing import Any

import cv2
import numpy as np

from pioneer_rknn import Yolo
from pioneer_sdk2 import Camera, CameraType, ImageViewer, ServoCamera


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_NAME = "qwert"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

OBJECT_THRESHOLD = 0.05
NMS_THRESHOLD = 0.45

STREAM_NAME = "boats_test"
STREAM_FPS = 10

SERVO_CAMERA_ANGLE = -80

ARUCO_DICTIONARY = "DICT_4X4_50"
MIN_CONFIRMATION_FRAMES = 3

# Порядок должен совпадать с data.yaml
CLASS_NAMES = {
    0: "green",
    1: "orange",
}

# Цвета OpenCV в формате BGR
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 165, 255),
}

PENDING_COLOR = (255, 255, 255)
ARUCO_COLOR = (255, 0, 255)
TEXT_COLOR = (0, 255, 255)


def prepare_input(
    frame: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Подготавливает изображение для RKNN.

    display_frame:
        BGR, (640, 640, 3) - для отображения.

    input_tensor:
        RGB, (1, 640, 640, 3), uint8 - для RKNN.
    """

    display_frame = cv2.resize(
        frame,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb_image = cv2.cvtColor(
        display_frame,
        cv2.COLOR_BGR2RGB,
    )

    input_tensor = np.expand_dims(
        rgb_image,
        axis=0,
    )

    input_tensor = np.ascontiguousarray(
        input_tensor,
        dtype=np.uint8,
    )

    return display_frame, input_tensor


def get_aruco_detector(dictionary_name: str) -> object:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install opencv-contrib-python."
        )

    if not hasattr(cv2.aruco, dictionary_name):
        available = ", ".join(
            name for name in dir(cv2.aruco) if name.startswith("DICT_")
        )
        raise ValueError(
            f"Unknown ArUco dictionary: {dictionary_name}. Available: {available}"
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name)
    )

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters


def get_aruco_center(corners: np.ndarray) -> tuple[float, float]:
    center_x = float(np.mean(corners[:, 0]))
    center_y = float(np.mean(corners[:, 1]))
    return center_x, center_y


def detect_aruco(frame: np.ndarray, detector: object) -> list[dict[str, Any]]:
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
        return []

    markers = []
    for marker_id, marker_corners in zip(ids.flatten(), corners):
        normalized_corners = np.asarray(
            marker_corners,
            dtype=np.float32,
        ).reshape(4, 2)

        markers.append(
            {
                "id": int(marker_id),
                "corners": normalized_corners,
                "center": get_aruco_center(normalized_corners),
            }
        )

    return markers


def normalize_yolo_detections(
    frame: np.ndarray,
    boxes: Any,
    classes: Any,
    scores: Any,
) -> list[dict[str, Any]]:
    if boxes is None or classes is None or scores is None:
        return []

    frame_height, frame_width = frame.shape[:2]
    detections = []

    for index, (box, class_id, score) in enumerate(zip(boxes, classes, scores)):
        class_id = int(class_id)
        score = float(score)

        x1, y1, x2, y2 = [
            int(round(float(value)))
            for value in box
        ]

        x1 = max(0, min(x1, frame_width - 1))
        y1 = max(0, min(y1, frame_height - 1))
        x2 = max(0, min(x2, frame_width - 1))
        y2 = max(0, min(y2, frame_height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            {
                "index": index,
                "box": (x1, y1, x2, y2),
                "class_id": class_id,
                "score": score,
                "area": float((x2 - x1) * (y2 - y1)),
            }
        )

    return detections


def match_markers_to_detections(
    markers: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    matches = {}

    for marker in markers:
        center_x, center_y = marker["center"]
        candidates = []

        for detection in detections:
            x1, y1, x2, y2 = detection["box"]
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                candidates.append(detection)

        if not candidates:
            continue

        best_detection = max(
            candidates,
            key=lambda item: (item["score"], -item["area"]),
        )

        matches[marker["id"]] = {
            "marker": marker,
            "detection": best_detection,
        }

    return matches


def create_vessel_entry(frame_number: int) -> dict[str, Any]:
    return {
        "green_score": 0.0,
        "orange_score": 0.0,
        "green_frames": 0,
        "orange_frames": 0,
        "total_matches": 0,
        "confirmed": False,
        "final_class_id": None,
        "best_score": 0.0,
        "first_seen_frame": frame_number,
        "last_seen_frame": frame_number,
        "last_box": None,
    }


def choose_final_class(vessel: dict[str, Any]) -> int | None:
    if vessel["green_score"] > vessel["orange_score"]:
        return 0

    if vessel["orange_score"] > vessel["green_score"]:
        return 1

    if vessel["green_frames"] > vessel["orange_frames"]:
        return 0

    if vessel["orange_frames"] > vessel["green_frames"]:
        return 1

    return vessel["final_class_id"]


def update_vessel_registry(
    vessels: dict[int, dict[str, Any]],
    matches: dict[int, dict[str, Any]],
    frame_number: int,
) -> None:
    for aruco_id, match in matches.items():
        detection = match["detection"]
        class_id = int(detection["class_id"])
        score = float(detection["score"])

        if aruco_id not in vessels:
            vessels[aruco_id] = create_vessel_entry(frame_number)

        vessel = vessels[aruco_id]
        vessel["total_matches"] += 1
        vessel["last_seen_frame"] = frame_number
        vessel["last_box"] = detection["box"]
        vessel["best_score"] = max(vessel["best_score"], score)

        if class_id == 0:
            vessel["green_score"] += score
            vessel["green_frames"] += 1
        elif class_id == 1:
            vessel["orange_score"] += score
            vessel["orange_frames"] += 1

        final_class_id = choose_final_class(vessel)
        if final_class_id is not None:
            vessel["final_class_id"] = final_class_id

        if vessel["total_matches"] >= MIN_CONFIRMATION_FRAMES:
            vessel["confirmed"] = vessel["final_class_id"] is not None


def get_vessel_statistics(
    vessels: dict[int, dict[str, Any]],
) -> dict[str, int]:
    green = 0
    orange = 0
    pending = 0

    for vessel in vessels.values():
        if not vessel["confirmed"]:
            pending += 1
            continue

        if vessel["final_class_id"] == 0:
            green += 1
        elif vessel["final_class_id"] == 1:
            orange += 1
        else:
            pending += 1

    return {
        "unique": green + orange,
        "green": green,
        "orange": orange,
        "pending": pending,
    }


def draw_aruco(
    frame: np.ndarray,
    markers: list[dict[str, Any]],
) -> None:
    for marker in markers:
        corners = marker["corners"].astype(np.int32)
        center_x, center_y = marker["center"]
        center = (int(round(center_x)), int(round(center_y)))

        cv2.polylines(
            frame,
            [corners],
            True,
            ARUCO_COLOR,
            2,
        )

        cv2.circle(
            frame,
            center,
            4,
            ARUCO_COLOR,
            -1,
        )

        cv2.putText(
            frame,
            f"ID:{marker['id']}",
            (center[0] + 6, max(20, center[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ARUCO_COLOR,
            2,
            cv2.LINE_AA,
        )


def get_vessel_label(
    aruco_id: int,
    detection: dict[str, Any],
    vessels: dict[int, dict[str, Any]],
) -> tuple[str, tuple[int, int, int]]:
    vessel = vessels.get(aruco_id)
    detection_class_id = int(detection["class_id"])

    if vessel is None or not vessel["confirmed"]:
        total_matches = 0 if vessel is None else vessel["total_matches"]
        return (
            f"ID:{aruco_id} pending {total_matches}/{MIN_CONFIRMATION_FRAMES}",
            PENDING_COLOR,
        )

    final_class_id = int(vessel["final_class_id"])
    class_name = CLASS_NAMES.get(final_class_id, f"class_{final_class_id}")
    color = CLASS_COLORS.get(final_class_id, PENDING_COLOR)
    return f"ID:{aruco_id} {class_name} confirmed", color


def draw_label(
    frame: np.ndarray,
    text: str,
    point: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    cv2.putText(
        frame,
        text,
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_detections_and_matches(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    matches: dict[int, dict[str, Any]],
    vessels: dict[int, dict[str, Any]],
) -> None:
    matched_ids_by_detection = {}
    for aruco_id, match in matches.items():
        detection_index = match["detection"]["index"]
        matched_ids_by_detection.setdefault(detection_index, []).append(aruco_id)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        class_id = int(detection["class_id"])
        score = float(detection["score"])
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        color = CLASS_COLORS.get(class_id, PENDING_COLOR)
        matched_ids = sorted(matched_ids_by_detection.get(detection["index"], []))

        if not matched_ids:
            label = f"{class_name}: {score:.2f}"
            label_color = color
        else:
            label = None
            label_color = color
            for aruco_id in matched_ids:
                vessel = vessels.get(aruco_id)
                if vessel is not None and vessel["confirmed"]:
                    label_color = CLASS_COLORS.get(
                        int(vessel["final_class_id"]),
                        color,
                    )
                    break

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            label_color,
            2,
        )

        if not matched_ids:
            draw_label(
                frame,
                label,
                (x1, max(25, y1 - 8)),
                label_color,
            )
            continue

        for offset, aruco_id in enumerate(matched_ids):
            vessel_label, vessel_color = get_vessel_label(
                aruco_id,
                detection,
                vessels,
            )
            draw_label(
                frame,
                vessel_label,
                (x1, max(25, y1 - 8 - offset * 22)),
                vessel_color,
            )


def draw_statistics(
    frame: np.ndarray,
    vessels: dict[int, dict[str, Any]],
    fps: float,
) -> None:
    stats = get_vessel_statistics(vessels)

    lines = [
        f"FPS: {fps:.1f}",
        f"Unique vessels: {stats['unique']}",
        f"Green: {stats['green']}",
        f"Orange: {stats['orange']}",
        f"Pending: {stats['pending']}",
    ]

    line_height = 24
    x = 15
    y = 28

    for line in lines:
        draw_label(frame, line, (x, y), TEXT_COLOR)
        y += line_height

    y += 8
    max_id_lines = max(0, (frame.shape[0] - y - 10) // line_height)

    for line_index, aruco_id in enumerate(sorted(vessels)):
        if line_index >= max_id_lines:
            draw_label(frame, "...", (x, y), TEXT_COLOR)
            break

        vessel = vessels[aruco_id]
        if vessel["confirmed"] and vessel["final_class_id"] is not None:
            class_id = int(vessel["final_class_id"])
            class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        else:
            class_name = "pending"

        draw_label(
            frame,
            f"ID {aruco_id}: {class_name}",
            (x, y),
            TEXT_COLOR,
        )
        y += line_height


def print_current_statistics(
    frame_number: int,
    vessels: dict[int, dict[str, Any]],
    inference_time: float,
    fps: float,
) -> None:
    stats = get_vessel_statistics(vessels)
    known_ids = ", ".join(str(aruco_id) for aruco_id in sorted(vessels))
    if not known_ids:
        known_ids = "-"

    print(
        f"Кадр {frame_number}: "
        f"unique={stats['unique']}, "
        f"green={stats['green']}, "
        f"orange={stats['orange']}, "
        f"pending={stats['pending']}, "
        f"время={inference_time * 1000:.1f} мс, "
        f"FPS={fps:.1f}, "
        f"ids={known_ids}"
    )


def print_final_report(vessels: dict[int, dict[str, Any]]) -> None:
    stats = get_vessel_statistics(vessels)

    print()
    print("========================================")
    print("ИТОГОВЫЙ ОТЧЁТ")
    print(f"Всего уникальных подтверждённых судов: {stats['unique']}")
    print(f"Green: {stats['green']}")
    print(f"Orange: {stats['orange']}")
    print(f"Неподтверждённых меток: {stats['pending']}")
    print()
    print("Суда:")

    if not vessels:
        print("Нет найденных судов")
    else:
        for aruco_id in sorted(vessels):
            vessel = vessels[aruco_id]
            if vessel["confirmed"] and vessel["final_class_id"] is not None:
                class_id = int(vessel["final_class_id"])
                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
            else:
                class_name = "pending"

            print(f"ArUco ID {aruco_id} — {class_name}")

    print("========================================")


def main() -> None:
    camera = None
    viewer = None
    model = None
    servo_camera = None
    vessels: dict[int, dict[str, Any]] = {}

    try:
        print("Инициализация камеры сервопривода...")
        servo_camera = ServoCamera()
        servo_result = servo_camera.set_angle(SERVO_CAMERA_ANGLE)
        print(f"Угол камеры: {SERVO_CAMERA_ANGLE}, результат: {servo_result}")

        print("Запуск камеры...")
        camera = Camera(
            camera_type=CameraType.MAIN,
        )

        viewer = ImageViewer()

        print("Создание ArUco-детектора...")
        aruco_detector = get_aruco_detector(ARUCO_DICTIONARY)
        print("Словарь ArUco:", ARUCO_DICTIONARY)

        print("Загрузка RKNN-модели...")
        model = Yolo(
            model_name=MODEL_NAME,
            object_thresh=OBJECT_THRESHOLD,
            nms_thresh=NMS_THRESHOLD,
            img_width=INPUT_WIDTH,
            img_height=INPUT_HEIGHT,
        )

        print("Модель загружена.")
        print("Модель:", MODEL_NAME)
        print("Порог:", OBJECT_THRESHOLD)
        print("Кадров для подтверждения:", MIN_CONFIRMATION_FRAMES)
        print()
        print("Открой поток:")
        print(f"http://10.42.0.1:8889/{STREAM_NAME}/")

        first_frame = True
        frame_number = 0

        while True:
            frame = camera.get_cv_frame(timeout=2.0)

            if frame is None:
                print("Кадр с камеры не получен")
                continue

            frame_number += 1
            display_frame, input_tensor = prepare_input(frame)

            if first_frame:
                print()
                print(
                    "Исходный кадр:",
                    frame.shape,
                    frame.dtype,
                )
                print(
                    "Вход RKNN:",
                    input_tensor.shape,
                    input_tensor.dtype,
                    "range:",
                    int(input_tensor.min()),
                    "...",
                    int(input_tensor.max()),
                )
                first_frame = False

            start_time = time.perf_counter()
            result = model.run([input_tensor])
            inference_time = time.perf_counter() - start_time

            fps = (
                1.0 / inference_time
                if inference_time > 0
                else 0.0
            )

            boxes = None
            classes = None
            scores = None

            if (
                isinstance(result, (tuple, list))
                and len(result) == 3
            ):
                boxes, classes, scores = result
            elif result is not None:
                print(
                    "Неожиданный формат результата:",
                    type(result),
                    repr(result),
                )

            detections = normalize_yolo_detections(
                display_frame,
                boxes,
                classes,
                scores,
            )

            try:
                markers = detect_aruco(display_frame, aruco_detector)
            except Exception as error:
                print(
                    f"Ошибка ArUco на кадре {frame_number}: "
                    f"{type(error).__name__}: {error}"
                )
                markers = []

            matches = match_markers_to_detections(
                markers,
                detections,
            )

            update_vessel_registry(
                vessels,
                matches,
                frame_number,
            )

            draw_aruco(
                display_frame,
                markers,
            )

            draw_detections_and_matches(
                display_frame,
                detections,
                matches,
                vessels,
            )

            draw_statistics(
                display_frame,
                vessels,
                fps,
            )

            if frame_number % 10 == 0:
                print_current_statistics(
                    frame_number,
                    vessels,
                    inference_time,
                    fps,
                )

            viewer.imshow(
                name=STREAM_NAME,
                frame=display_frame,
                fps=STREAM_FPS,
            )

    except KeyboardInterrupt:
        print("\nОстановлено пользователем")

    except Exception as error:
        print(
            f"\nОшибка: "
            f"{type(error).__name__}: {error}"
        )
        raise

    finally:
        print_final_report(vessels)
        print("Освобождение ресурсов...")

        if model is not None:
            try:
                model.release()
            except Exception as error:
                print(
                    "Ошибка освобождения модели:",
                    error,
                )

        if camera is not None:
            try:
                camera.stop()
            except Exception as error:
                print(
                    "Ошибка остановки камеры:",
                    error,
                )

        if viewer is not None:
            try:
                viewer.close()
            except Exception as error:
                print(
                    "Ошибка закрытия потока:",
                    error,
                )

        if servo_camera is not None and hasattr(servo_camera, "close"):
            try:
                servo_camera.close()
            except Exception as error:
                print(
                    "Ошибка закрытия сервокамеры:",
                    error,
                )

        print("Готово")


if __name__ == "__main__":
    main()
