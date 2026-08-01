from __future__ import annotations

import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request


# ============================================================
# НАСТРОЙКИ
# ============================================================

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001

FIELD_WIDTH_M = 8.0
FIELD_HEIGHT_M = 6.0

# Локальная координата (0, 0) находится на карте в (1.5, 4.5).
DRONE_START_MAP_X = 1.5
DRONE_START_MAP_Y = 4.5

MAP_IMAGE_PATH = Path(
    r"C:\Users\1234\arh26\3_track\image.jpg"
)

WINDOW_NAME = "Geoscan Archipelag 2026 - Control Station"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

POSE_PATH_MIN_STEP_M = 0.03
MAX_POSE_PATH_POINTS = 5000


# ============================================================
# ЦВЕТА BGR
# ============================================================

COLOR_MAP = {
    "исправна": (0, 255, 0),
    "покрыта пылью": (0, 165, 255),
    "неисправна": (0, 0, 255),
    "обнаружена": (0, 255, 255),
    "panel": (0, 255, 0),
}

UNKNOWN_COLOR = (255, 0, 255)
POSE_COLOR = (255, 0, 0)
PATH_COLOR = (255, 255, 0)
START_COLOR = (255, 0, 255)

STATUS_LABELS = {
    "исправна": "OK",
    "покрыта пылью": "DUST",
    "неисправна": "FAULT",
    "обнаружена": "DETECTED",
    "panel": "PANEL",
}


# ============================================================
# FLASK И ОБЩЕЕ СОСТОЯНИЕ
# ============================================================

app = Flask(__name__)

data_lock = threading.Lock()
stop_event = threading.Event()
map_needs_update = threading.Event()
map_needs_update.set()

detected_stations: dict[int, dict] = {}

current_pose: dict = {
    "x": 0.0,
    "y": 0.0,
    "valid": False,
    "tracked_features": 0,
    "inliers": 0,
    "flow_u": 0.0,
    "flow_v": 0.0,
    "rotation_deg": 0.0,
    "total_distance": 0.0,
    "coordinate_mode": "visual_odometry",
    "received_at": 0.0,
}

pose_path: list[tuple[float, float]] = [(0.0, 0.0)]


# ============================================================
# КООРДИНАТЫ
# ============================================================


def drone_to_map_coordinates(
    x_drone: float,
    y_drone: float,
) -> tuple[float, float]:
    """
    Локальная система дрона:
        +X направлен вниз по карте;
        +Y направлен влево по карте.

    Система карты:
        map X растёт справа налево;
        map Y растёт сверху вниз.

    Поэтому:
        map_x = 1.5 + local_y
        map_y = 4.5 + local_x
    """

    return (
        DRONE_START_MAP_X + y_drone,
        DRONE_START_MAP_Y + x_drone,
    )


def map_meters_to_pixels(
    x_m: float,
    y_m: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """
    На карте (0, 0) находится сверху справа.
    X растёт справа налево, Y — сверху вниз.
    """

    pixel_x = int(round(
        (1.0 - x_m / FIELD_WIDTH_M)
        * (image_width - 1)
    ))
    pixel_y = int(round(
        (y_m / FIELD_HEIGHT_M)
        * (image_height - 1)
    ))

    return pixel_x, pixel_y


def point_inside_field(x_map: float, y_map: float) -> bool:
    return (
        0.0 <= x_map <= FIELD_WIDTH_M
        and 0.0 <= y_map <= FIELD_HEIGHT_M
    )


# ============================================================
# ПРОВЕРКА JSON
# ============================================================


def finite_float(data: dict, name: str) -> float:
    value = float(data[name])

    if not math.isfinite(value):
        raise ValueError(f"Поле {name} должно быть конечным числом")

    return value


def optional_float(data: dict, name: str) -> float | None:
    value = data.get(name)

    if value is None:
        return None

    parsed = float(value)

    if not math.isfinite(parsed):
        raise ValueError(f"Поле {name} должно быть конечным числом")

    return parsed


def parse_station_data(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Тело запроса должно быть JSON-объектом")

    missing = {"id", "x", "y"} - data.keys()

    if missing:
        raise ValueError(
            "Отсутствуют поля: " + ", ".join(sorted(missing))
        )

    station_id = int(data["id"])

    if station_id <= 0:
        raise ValueError("id должен быть положительным")

    return {
        "id": station_id,
        "x": finite_float(data, "x"),
        "y": finite_float(data, "y"),
        "status": str(data.get("status", "обнаружена")),
        "confidence": optional_float(data, "confidence"),
        "confirmations": int(data.get("confirmations", 1)),
        "dust_votes": int(data.get("dust_votes", 0)),
        "clean_votes": int(data.get("clean_votes", 0)),
        "coordinate_mode": str(
            data.get("coordinate_mode", "unknown")
        ),
        "received_at": time.time(),
    }


def parse_pose_data(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Тело запроса должно быть JSON-объектом")

    missing = {"x", "y"} - data.keys()

    if missing:
        raise ValueError(
            "Отсутствуют поля: " + ", ".join(sorted(missing))
        )

    return {
        "x": finite_float(data, "x"),
        "y": finite_float(data, "y"),
        "valid": bool(data.get("valid", False)),
        "tracked_features": int(data.get("tracked_features", 0)),
        "inliers": int(data.get("inliers", 0)),
        "flow_u": float(data.get("flow_u", 0.0)),
        "flow_v": float(data.get("flow_v", 0.0)),
        "rotation_deg": float(data.get("rotation_deg", 0.0)),
        "total_distance": float(data.get("total_distance", 0.0)),
        "coordinate_mode": str(
            data.get("coordinate_mode", "visual_odometry")
        ),
        "received_at": time.time(),
    }


# ============================================================
# HTTP-МАРШРУТЫ
# ============================================================


@app.route("/drone", methods=["POST"])
def receive_station():
    try:
        station = parse_station_data(
            request.get_json(force=True)
        )
    except Exception as error:
        print("[ОШИБКА СТАНЦИИ]", type(error).__name__, error)
        return jsonify({
            "status": "error",
            "message": str(error),
        }), 400

    station_id = station["id"]

    with data_lock:
        is_new = station_id not in detected_stations
        detected_stations[station_id] = station
        station_count = len(detected_stations)

    map_needs_update.set()

    print(
        f"[HTTP] Станция №{station_id} "
        f"{'добавлена' if is_new else 'обновлена'}: "
        f"local=({station['x']:.3f}, {station['y']:.3f}), "
        f"status={station['status']}, "
        f"n={station['confirmations']}"
    )

    return jsonify({
        "status": "ok",
        "station_id": station_id,
        "stations_count": station_count,
    }), 200


@app.route("/pose", methods=["POST"])
def receive_pose():
    try:
        pose = parse_pose_data(
            request.get_json(force=True)
        )
    except Exception as error:
        print("[ОШИБКА ПОЗЫ]", type(error).__name__, error)
        return jsonify({
            "status": "error",
            "message": str(error),
        }), 400

    with data_lock:
        current_pose.update(pose)

        last_x, last_y = pose_path[-1]
        moved = math.hypot(
            pose["x"] - last_x,
            pose["y"] - last_y,
        )

        if moved >= POSE_PATH_MIN_STEP_M:
            pose_path.append((pose["x"], pose["y"]))

            if len(pose_path) > MAX_POSE_PATH_POINTS:
                del pose_path[: len(pose_path) - MAX_POSE_PATH_POINTS]

    map_needs_update.set()

    return jsonify({
        "status": "ok",
        "x": pose["x"],
        "y": pose["y"],
    }), 200


@app.route("/stations", methods=["GET"])
def get_stations():
    with data_lock:
        stations = [dict(item) for item in detected_stations.values()]
        pose = dict(current_pose)
        path = list(pose_path)

    return jsonify({
        "count": len(stations),
        "stations": stations,
        "pose": pose,
        "path": path,
    }), 200


@app.route("/clear", methods=["POST"])
def clear_state():
    with data_lock:
        detected_stations.clear()
        pose_path.clear()
        pose_path.append((0.0, 0.0))
        current_pose.update({
            "x": 0.0,
            "y": 0.0,
            "valid": False,
            "tracked_features": 0,
            "inliers": 0,
            "flow_u": 0.0,
            "flow_v": 0.0,
            "rotation_deg": 0.0,
            "total_distance": 0.0,
            "coordinate_mode": "visual_odometry",
            "received_at": time.time(),
        })

    map_needs_update.set()
    print("[HTTP] Карта, путь и поза очищены")

    return jsonify({
        "status": "ok",
        "message": "state cleared",
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "server": "map_manual_tracking",
    }), 200


def run_http_server() -> None:
    print(
        f"[СТАРТ] HTTP-сервер: "
        f"http://0.0.0.0:{SERVER_PORT}"
    )

    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# ============================================================
# ОТРИСОВКА
# ============================================================


def draw_text_with_background(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.52,
) -> None:
    x, y = position
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        thickness,
    )

    x = max(4, min(x, frame.shape[1] - text_width - 8))
    y = max(text_height + 8, min(y, frame.shape[0] - baseline - 6))

    cv2.rectangle(
        frame,
        (x - 4, y - text_height - 6),
        (x + text_width + 4, y + baseline + 4),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def draw_meter_grid(
    frame: np.ndarray,
    step_m: float = 1.0,
) -> None:
    height, width = frame.shape[:2]
    overlay = frame.copy()

    grid_color = (175, 175, 175)
    axis_color = (255, 255, 255)

    x_m = 0.0
    while x_m <= FIELD_WIDTH_M + 1e-6:
        pixel_x, _ = map_meters_to_pixels(
            x_m,
            0.0,
            width,
            height,
        )
        pixel_x = max(0, min(pixel_x, width - 1))

        cv2.line(
            overlay,
            (pixel_x, 0),
            (pixel_x, height - 1),
            axis_color if abs(x_m) < 1e-6 else grid_color,
            2 if abs(x_m) < 1e-6 else 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"{x_m:g}",
            (max(2, pixel_x - 7), 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        x_m += step_m

    y_m = 0.0
    while y_m <= FIELD_HEIGHT_M + 1e-6:
        _, pixel_y = map_meters_to_pixels(
            0.0,
            y_m,
            width,
            height,
        )
        pixel_y = max(0, min(pixel_y, height - 1))

        cv2.line(
            overlay,
            (0, pixel_y),
            (width - 1, pixel_y),
            axis_color if abs(y_m) < 1e-6 else grid_color,
            2 if abs(y_m) < 1e-6 else 1,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"{y_m:g}",
            (5, max(15, pixel_y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y_m += step_m

    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


def local_point_to_pixel(
    x_local: float,
    y_local: float,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    x_map, y_map = drone_to_map_coordinates(x_local, y_local)

    if not point_inside_field(x_map, y_map):
        return None

    return map_meters_to_pixels(
        x_map,
        y_map,
        width,
        height,
    )


def draw_start(frame: np.ndarray) -> None:
    height, width = frame.shape[:2]
    point = local_point_to_pixel(0.0, 0.0, width, height)

    if point is None:
        return

    cv2.circle(frame, point, 14, (0, 0, 0), -1)
    cv2.circle(frame, point, 10, START_COLOR, -1)
    cv2.circle(frame, point, 3, (255, 255, 255), -1)

    draw_text_with_background(
        frame,
        "START local=(0,0) map=(1.5,4.5)",
        (point[0] + 16, point[1] - 10),
        0.48,
    )


def draw_pose_path(
    frame: np.ndarray,
    pose: dict,
    path: list[tuple[float, float]],
) -> None:
    height, width = frame.shape[:2]

    pixel_path: list[tuple[int, int]] = []

    for x_local, y_local in path:
        point = local_point_to_pixel(
            x_local,
            y_local,
            width,
            height,
        )
        if point is not None:
            pixel_path.append(point)

    if len(pixel_path) >= 2:
        cv2.polylines(
            frame,
            [np.asarray(pixel_path, dtype=np.int32)],
            False,
            PATH_COLOR,
            2,
            cv2.LINE_AA,
        )

    current_point = local_point_to_pixel(
        float(pose["x"]),
        float(pose["y"]),
        width,
        height,
    )

    if current_point is None:
        return

    cv2.circle(frame, current_point, 16, (0, 0, 0), -1)
    cv2.circle(frame, current_point, 12, POSE_COLOR, -1)
    cv2.circle(frame, current_point, 3, (255, 255, 255), -1)

    odom_label = "ODOM OK" if pose.get("valid") else "ODOM WAIT"
    draw_text_with_background(
        frame,
        (
            f"{odom_label} local=({pose['x']:.2f},"
            f"{pose['y']:.2f})"
        ),
        (current_point[0] + 18, current_point[1] - 12),
        0.5,
    )


def draw_station(frame: np.ndarray, station: dict) -> None:
    height, width = frame.shape[:2]

    x_local = float(station["x"])
    y_local = float(station["y"])
    point = local_point_to_pixel(x_local, y_local, width, height)

    if point is None:
        x_map, y_map = drone_to_map_coordinates(x_local, y_local)
        print(
            f"[ПРЕДУПРЕЖДЕНИЕ] Станция №{station['id']} "
            f"вне карты: map=({x_map:.2f}, {y_map:.2f})"
        )
        return

    status = str(station.get("status", "обнаружена"))
    color = COLOR_MAP.get(status, UNKNOWN_COLOR)
    status_label = STATUS_LABELS.get(status, status)

    cv2.circle(frame, point, 15, (0, 0, 0), -1)
    cv2.circle(frame, point, 11, color, -1)
    cv2.circle(frame, point, 3, (255, 255, 255), -1)

    confidence = station.get("confidence")
    confirmations = int(station.get("confirmations", 1))
    dust_votes = int(station.get("dust_votes", 0))
    clean_votes = int(station.get("clean_votes", 0))
    mode = str(station.get("coordinate_mode", "unknown"))

    first_line = f"#{station['id']} {status_label}"
    second_line = f"local=({x_local:.2f},{y_local:.2f})"

    if confidence is not None:
        second_line += f" conf={float(confidence):.2f}"

    third_line = (
        f"n={confirmations} dust={dust_votes} "
        f"clean={clean_votes}"
    )
    fourth_line = f"mode={mode}"

    label_x = point[0] + 18
    label_y = point[1] - 28

    draw_text_with_background(frame, first_line, (label_x, label_y), 0.58)
    draw_text_with_background(frame, second_line, (label_x, label_y + 22), 0.48)
    draw_text_with_background(frame, third_line, (label_x, label_y + 44), 0.46)
    draw_text_with_background(frame, fourth_line, (label_x, label_y + 66), 0.44)


def draw_header(
    frame: np.ndarray,
    stations_count: int,
    pose: dict,
) -> None:
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (470, 120),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)

    lines = [
        "Geoscan Archipelag 2026",
        f"Stations: {stations_count}",
        (
            f"Pose local=({pose['x']:.2f}, {pose['y']:.2f}) "
            f"inliers={pose.get('inliers', 0)}"
        ),
        f"Distance: {pose.get('total_distance', 0.0):.2f} m",
    ]

    y = 36
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (22, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if index == 0 else 0.55,
            (255, 255, 255),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
        y += 25


def build_map_frame(original_map: np.ndarray) -> np.ndarray:
    output = original_map.copy()

    with data_lock:
        stations = [
            dict(station)
            for station in detected_stations.values()
        ]
        pose = dict(current_pose)
        path = list(pose_path)

    draw_meter_grid(output, step_m=1.0)
    draw_start(output)
    draw_pose_path(output, pose, path)

    for station in stations:
        draw_station(output, station)

    draw_header(output, len(stations), pose)
    return output


# ============================================================
# ОКНО
# ============================================================


def display_loop() -> None:
    original_map = cv2.imread(str(MAP_IMAGE_PATH))

    if original_map is None:
        raise FileNotFoundError(
            f"Не удалось открыть карту: {MAP_IMAGE_PATH}"
        )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)

    working_map = build_map_frame(original_map)

    print("Окно карты запущено")
    print("ESC — завершить")
    print("C — очистить станции, путь и позу")
    print("S — сохранить изображение")

    while not stop_event.is_set():
        if map_needs_update.is_set():
            working_map = build_map_frame(original_map)
            map_needs_update.clear()

        cv2.imshow(WINDOW_NAME, working_map)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            stop_event.set()
            break

        if key in (ord("c"), ord("C")):
            with data_lock:
                detected_stations.clear()
                pose_path.clear()
                pose_path.append((0.0, 0.0))
                current_pose.update({
                    "x": 0.0,
                    "y": 0.0,
                    "valid": False,
                    "tracked_features": 0,
                    "inliers": 0,
                    "flow_u": 0.0,
                    "flow_v": 0.0,
                    "rotation_deg": 0.0,
                    "total_distance": 0.0,
                    "coordinate_mode": "visual_odometry",
                    "received_at": time.time(),
                })

            map_needs_update.set()
            print("[КАРТА] Состояние очищено")

        if key in (ord("s"), ord("S")):
            output_path = Path("detected_stations_map.jpg")
            cv2.imwrite(str(output_path), working_map)
            print(f"[КАРТА] Сохранено: {output_path.resolve()}")

    cv2.destroyAllWindows()


# ============================================================
# ЗАПУСК
# ============================================================


def main() -> None:
    if not MAP_IMAGE_PATH.is_file():
        raise FileNotFoundError(
            f"Карта не найдена: {MAP_IMAGE_PATH}"
        )

    server_thread = threading.Thread(
        target=run_http_server,
        daemon=True,
    )
    server_thread.start()

    try:
        display_loop()
    except KeyboardInterrupt:
        print("\nПрограмма остановлена")
    finally:
        stop_event.set()
        cv2.destroyAllWindows()
        print("Принимающая сторона остановлена")


if __name__ == "__main__":
    main()