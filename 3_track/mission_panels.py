from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pioneer_sdk2
import requests
from flask import Flask, jsonify
from pioneer_rknn import Yolo
from pioneer_sdk2 import Camera, CameraType, ImageViewer, Pioneer, ServoCamera


# ============================================================
# СЕТЬ
# ============================================================

# На этом адресе дрон ждёт POST /start.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001

# Адрес компьютера, где запущен map.py с маршрутом POST /drone.
# Впиши IPv4 компьютера в сети Pioneer.
CLIENT_URL = "http://172.17.49.57:5001/drone"

HTTP_TIMEOUT = 1.5
HTTP_RETRIES = 3


# ============================================================
# ПОЛЁТ
# ============================================================

FLIGHT_HEIGHT = 1.3

# x, y, z, yaw, time
ROUTE = [
    (0.5, 4.0, FLIGHT_HEIGHT, 0.0, 0),
    (-1.5, 4.0, FLIGHT_HEIGHT, 0.0, 0),
    (-1.5, 2.0, FLIGHT_HEIGHT, 0.0, 0),
]

LANDING_X = -0.2
LANDING_Y = -0.65
RETURN_HEIGHT = 0.8
PRELAND_HEIGHT = 0.3
POINT_TIMEOUT = 30.0

# После взлёта и после каждой маршрутной точки дрон зависает,
# чтобы накопить несколько подтверждений панели.
SCAN_SECONDS = 2.5


# ============================================================
# КАМЕРА И YOLO
# ============================================================

CAMERA_ANGLE = -80
MODEL_NAME = "rknn_3576"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

OBJECT_THRESHOLD = 0.85
NMS_THRESHOLD = 0.45

STREAM_NAME = "video"
STREAM_FPS = 15

CLASS_NAMES = {
    0: "dust",
    1: "panel",
}

CLASS_COLORS = {
    0: (0, 165, 255),
    1: (0, 255, 0),
}

STATION_CLASS_IDS = { 1}
UNKNOWN_COLOR = (255, 255, 255)


# ============================================================
# ПЕРЕВОД ЦЕНТРА БОКСА В СМЕЩЕНИЕ ОТ ДРОНА
# ============================================================

# Пока гомография не заполнена, координата панели принимается
# равной координате дрона, но только если бокс близко к центру кадра.
CENTER_ACCEPT_RADIUS_PX = 70.0

# После калибровки вставь сюда четыре пары точек.
# Порядок точек в двух массивах должен совпадать.
#
# IMAGE_CALIBRATION_POINTS = np.float32([
#     [u1, v1],
#     [u2, v2],
#     [u3, v3],
#     [u4, v4],
# ])
#
# BODY_CALIBRATION_POINTS = np.float32([
#     [dx1, dy1],
#     [dx2, dy2],
#     [dx3, dy3],
#     [dx4, dy4],
# ])
#
# u/v — пиксели на обработанном кадре 640x640.
# dx/dy — реальные смещения от дрона по его локальным X/Y, в метрах.
IMAGE_CALIBRATION_POINTS: np.ndarray | None = None
BODY_CALIBRATION_POINTS: np.ndarray | None = None


def build_pixel_to_body_homography() -> np.ndarray | None:
    if IMAGE_CALIBRATION_POINTS is None or BODY_CALIBRATION_POINTS is None:
        return None

    image_points = np.asarray(IMAGE_CALIBRATION_POINTS, dtype=np.float32)
    body_points = np.asarray(BODY_CALIBRATION_POINTS, dtype=np.float32)

    if image_points.shape != (4, 2) or body_points.shape != (4, 2):
        raise ValueError("Калибровочные массивы должны иметь форму (4, 2)")

    return cv2.getPerspectiveTransform(image_points, body_points)


PIXEL_TO_BODY_H = build_pixel_to_body_homography()


# ============================================================
# ОБЪЕДИНЕНИЕ ПОВТОРНЫХ ДЕТЕКЦИЙ
# ============================================================

STATION_MERGE_RADIUS_M = 0.45
MIN_STATION_CONFIRMATIONS = 4
DETECTION_PRINT_INTERVAL = 1.0


# ============================================================
# СОСТОЯНИЕ
# ============================================================

app = Flask(__name__)

point_event = threading.Event()
start_event = threading.Event()
vision_stop_event = threading.Event()
measurement_enabled = threading.Event()

state_lock = threading.Lock()
pose_lock = threading.Lock()
stations_lock = threading.Lock()

mission_started = False
vision_running = False
airborne = False

processed_frames = 0
last_inference_ms = 0.0
last_detections: list[dict] = []

# Позиция считается известной во время зависания после POINT_REACHED.
stable_drone_pose: tuple[float, float, float] | None = None

station_tracks: list[dict] = []
next_station_id = 1


drone: Pioneer | None = None
camera: Camera | None = None
viewer: ImageViewer | None = None
servo_camera: ServoCamera | None = None
model: Yolo | None = None
vision_thread: threading.Thread | None = None


# ============================================================
# HTTP-СЕРВЕР ДРОНА
# ============================================================

@app.route("/start", methods=["POST"])
def receive_start_command():
    with state_lock:
        if mission_started or start_event.is_set():
            return jsonify({
                "status": "error",
                "message": "mission already started",
            }), 409

        start_event.set()

    print("Получена команда POST /start")

    return jsonify({
        "status": "accepted",
        "message": "start command received",
    }), 202


@app.route("/status", methods=["GET"])
def get_status():
    with state_lock:
        result = {
            "mission_started": mission_started,
            "vision_running": vision_running,
            "processed_frames": processed_frames,
            "inference_ms": round(last_inference_ms, 2),
            "detections": list(last_detections),
            "measurement_enabled": measurement_enabled.is_set(),
            "stream": f"http://172.17.49.101:8889/{STREAM_NAME}/",
        }

    with stations_lock:
        result["stations"] = [
            {
                "id": track["id"],
                "x": round(track["x"], 3),
                "y": round(track["y"], 3),
                "status": track["status"],
                "confirmations": track["count"],
                "sent": track["sent"],
            }
            for track in station_tracks
        ]

    return jsonify(result), 200


def run_http_server():
    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )


# ============================================================
# ПОЛЁТ И ОКНА СКАНИРОВАНИЯ
# ============================================================


def point_reached(event):
    point_event.set()


def wait_for_point(timeout: float = POINT_TIMEOUT):
    if not point_event.wait(timeout=timeout):
        raise TimeoutError(f"Дрон не достиг точки за {timeout} секунд")

    point_event.clear()


def scan_at_known_pose(x: float, y: float, z: float):
    global stable_drone_pose

    with pose_lock:
        stable_drone_pose = (float(x), float(y), float(z))

    measurement_enabled.set()

    print(
        f"Сканирование: x={x:.2f}, y={y:.2f}, "
        f"z={z:.2f}, {SCAN_SECONDS:.1f} с"
    )

    time.sleep(SCAN_SECONDS)
    measurement_enabled.clear()


def fly_to_point(
    drone_object: Pioneer,
    x: float,
    y: float,
    z: float,
    yaw: float,
    duration: int,
    scan_after: bool = True,
):
    measurement_enabled.clear()
    point_event.clear()

    print(
        f"Летим в точку: x={x}, y={y}, z={z}, "
        f"yaw={yaw}, time={duration}"
    )

    if drone_object.go_to_local_point(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        time=duration,
    ) is False:
        raise RuntimeError(f"Команда перехода в точку x={x}, y={y}, z={z} отклонена")

    wait_for_point()
    print("Точка достигнута")

    if scan_after:
        scan_at_known_pose(x, y, z)


# ============================================================
# ГЕОМЕТРИЯ
# ============================================================


def clamp_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    values = np.asarray(box, dtype=np.float32).reshape(-1)

    if values.size < 4:
        return None

    x1, y1, x2, y2 = values[:4]

    x1 = max(0, min(int(round(float(x1))), width - 1))
    y1 = max(0, min(int(round(float(y1))), height - 1))
    x2 = max(0, min(int(round(float(x2))), width - 1))
    y2 = max(0, min(int(round(float(y2))), height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def pixel_to_body_offset(u: float, v: float) -> tuple[float, float] | None:
    if PIXEL_TO_BODY_H is None:
        return None

    point = np.array([[[u, v]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(point, PIXEL_TO_BODY_H)
    dx, dy = transformed[0, 0]

    return float(dx), float(dy)


def estimate_station_coordinates(
    center_x: float,
    center_y: float,
) -> tuple[float, float, str] | None:
    if not measurement_enabled.is_set():
        return None

    with pose_lock:
        pose = stable_drone_pose

    if pose is None:
        return None

    drone_x, drone_y, _ = pose
    offset = pixel_to_body_offset(center_x, center_y)

    if offset is not None:
        dx, dy = offset
        return drone_x + dx, drone_y + dy, "homography"

    distance_to_center = float(np.hypot(
        center_x - INPUT_WIDTH / 2.0,
        center_y - INPUT_HEIGHT / 2.0,
    ))

    if distance_to_center <= CENTER_ACCEPT_RADIUS_PX:
        return drone_x, drone_y, "center"

    return None


# ============================================================
# ОТПРАВКА НА КОМПЬЮТЕР
# ============================================================


def send_station_payload(track_id: int, payload: dict):
    success = False

    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            response = requests.post(
                CLIENT_URL,
                json=payload,
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()

            print(
                f"[HTTP] Станция №{track_id} отправлена: "
                f"x={payload['x']:.3f}, y={payload['y']:.3f}, "
                f"status={payload['status']}"
            )

            success = True
            break

        except requests.RequestException as error:
            print(
                f"[HTTP] Станция №{track_id}, "
                f"попытка {attempt}/{HTTP_RETRIES}: {error}"
            )

            if attempt < HTTP_RETRIES:
                time.sleep(0.5)

    with stations_lock:
        for track in station_tracks:
            if track["id"] == track_id:
                track["sent"] = success
                track["sending"] = False
                break


def register_station_observation(
    x: float,
    y: float,
    status: str,
    confidence: float,
    coordinate_mode: str,
):
    global next_station_id

    payload_to_send: tuple[int, dict] | None = None

    with stations_lock:
        nearest_track = None
        nearest_distance = float("inf")

        for track in station_tracks:
            distance = float(np.hypot(track["x"] - x, track["y"] - y))

            if distance < nearest_distance:
                nearest_distance = distance
                nearest_track = track

        if nearest_track is None or nearest_distance > STATION_MERGE_RADIUS_M:
            nearest_track = {
                "id": next_station_id,
                "x": float(x),
                "y": float(y),
                "count": 1,
                "confidence_sum": float(confidence),
                "status": status,
                "coordinate_mode": coordinate_mode,
                "sent": False,
                "sending": False,
            }
            station_tracks.append(nearest_track)
            next_station_id += 1

            print(
                f"Новый кандидат станции №{nearest_track['id']}: "
                f"x={x:.2f}, y={y:.2f}, status={status}, "
                f"mode={coordinate_mode}"
            )

        else:
            old_count = nearest_track["count"]
            new_count = old_count + 1

            nearest_track["x"] = (nearest_track["x"] * old_count + x) / new_count
            nearest_track["y"] = (nearest_track["y"] * old_count + y) / new_count
            nearest_track["count"] = new_count
            nearest_track["confidence_sum"] += confidence

            if status == "dust":
                nearest_track["status"] = "dust"

            if coordinate_mode == "homography":
                nearest_track["coordinate_mode"] = "homography"

        if (
            nearest_track["count"] >= MIN_STATION_CONFIRMATIONS
            and not nearest_track["sent"]
            and not nearest_track["sending"]
        ):
            nearest_track["sending"] = True

            payload = {
                "id": int(nearest_track["id"]),
                "x": round(float(nearest_track["x"]), 3),
                "y": round(float(nearest_track["y"]), 3),
                "status": str(nearest_track["status"]),
                "confidence": round(
                    float(nearest_track["confidence_sum"] / nearest_track["count"]),
                    3,
                ),
                "confirmations": int(nearest_track["count"]),
                "coordinate_mode": str(nearest_track["coordinate_mode"]),
            }

            payload_to_send = int(nearest_track["id"]), payload

    if payload_to_send is not None:
        threading.Thread(
            target=send_station_payload,
            args=payload_to_send,
            daemon=True,
        ).start()


# ============================================================
# ОТРИСОВКА ДЕТЕКЦИЙ
# ============================================================


def draw_label(frame, text: str, x: int, y: int, color):
    cv2.putText(
        frame,
        text,
        (x, max(20, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, max(20, y - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_detections(frame, result, last_print_times: dict[int, float]) -> list[dict]:
    detections: list[dict] = []

    if result is None:
        return detections

    if not isinstance(result, (tuple, list)) or len(result) != 3:
        print("Неожиданный результат YOLO:", type(result))
        return detections

    boxes, classes, scores = result

    if boxes is None or classes is None or scores is None:
        return detections

    height, width = frame.shape[:2]

    for box, class_id, score in zip(boxes, classes, scores):
        class_id = int(class_id)
        confidence = float(score)

        normalized = clamp_box(box, width, height)
        if normalized is None:
            continue

        x1, y1, x2, y2 = normalized
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        color = CLASS_COLORS.get(class_id, UNKNOWN_COLOR)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, f"{class_name} {confidence:.2f}", x1, y1, color)

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0

        station_coordinates = None
        coordinate_text = "not-localized"

        if class_id in STATION_CLASS_IDS:
            station_coordinates = estimate_station_coordinates(center_x, center_y)

            if station_coordinates is not None:
                station_x, station_y, coordinate_mode = station_coordinates
                coordinate_text = (
                    f"local=({station_x:.2f},{station_y:.2f}) {coordinate_mode}"
                )

                register_station_observation(
                    x=station_x,
                    y=station_y,
                    status=class_name,
                    confidence=confidence,
                    coordinate_mode=coordinate_mode,
                )

            cx = int(round(center_x))
            cy = int(round(center_y))
            cv2.drawMarker(
                frame,
                (cx, cy),
                (255, 255, 255),
                cv2.MARKER_CROSS,
                18,
                2,
                cv2.LINE_AA,
            )
            draw_label(
                frame,
                f"px=({center_x:.1f},{center_y:.1f}) {coordinate_text}",
                min(cx + 10, width - 300),
                max(cy - 10, 20),
                (255, 255, 255),
            )

        detection = {
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "box": [x1, y1, x2, y2],
            "center": [round(center_x, 1), round(center_y, 1)],
        }

        if station_coordinates is not None:
            detection["local_x"] = round(station_coordinates[0], 3)
            detection["local_y"] = round(station_coordinates[1], 3)
            detection["coordinate_mode"] = station_coordinates[2]

        detections.append(detection)

        now = time.monotonic()
        if now - last_print_times.get(class_id, 0.0) >= DETECTION_PRINT_INTERVAL:
            print(
                f"Обнаружено: {class_name}, confidence={confidence:.3f}, "
                f"center=({center_x:.1f}, {center_y:.1f})"
            )
            last_print_times[class_id] = now

    return detections


def draw_statistics(frame, inference_ms: float, detections_count: int):
    fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0

    with stations_lock:
        stations_count = len(station_tracks)
        sent_count = sum(1 for track in station_tracks if track["sent"])

    lines = [
        f"FPS: {fps:.1f}",
        f"Inference: {inference_ms:.1f} ms",
        f"Objects: {detections_count}",
        f"Stations: {stations_count}, sent: {sent_count}",
        "Coordinate scan: ON" if measurement_enabled.is_set() else "Coordinate scan: OFF",
    ]

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (330, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (16, 28 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


# ============================================================
# ПОТОК КАМЕРЫ И NPU
# ============================================================


def vision_loop():
    global vision_running, processed_frames, last_inference_ms, last_detections

    if camera is None or viewer is None or model is None:
        raise RuntimeError("Камера, ImageViewer или модель не инициализированы")

    frame_number = 0
    last_print_times: dict[int, float] = {}

    with state_lock:
        vision_running = True

    print("Обработка камеры и RKNN запущена")
    print(f"Трансляция: http://172.17.49.101:8889/{STREAM_NAME}/")

    try:
        while not vision_stop_event.is_set():
            frame = camera.get_cv_frame(timeout=1.0)
            if frame is None:
                continue

            display_frame = cv2.resize(
                frame,
                (INPUT_WIDTH, INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(rgb, axis=0)
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)

            started = time.perf_counter()
            result = model.run([input_tensor])
            inference_ms = (time.perf_counter() - started) * 1000.0

            detections = draw_detections(display_frame, result, last_print_times)
            draw_statistics(display_frame, inference_ms, len(detections))

            viewer.imshow(
                name=STREAM_NAME,
                frame=display_frame,
                fps=STREAM_FPS,
            )

            frame_number += 1

            with state_lock:
                processed_frames = frame_number
                last_inference_ms = inference_ms
                last_detections = detections

            if frame_number % 30 == 0:
                print(
                    f"Кадр {frame_number} | inference={inference_ms:.1f} мс | "
                    f"объектов={len(detections)}"
                )

    except Exception as error:
        print("Ошибка в потоке камеры:", type(error).__name__, error)

    finally:
        with state_lock:
            vision_running = False

        print("Обработка камеры и RKNN остановлена")


def stop_vision():
    global vision_thread

    vision_stop_event.set()
    measurement_enabled.clear()

    if vision_thread is not None and vision_thread.is_alive():
        vision_thread.join(timeout=3.0)


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

try:
    print("Подключение к автопилоту")

    drone = Pioneer()
    drone.subscribe(point_reached, pioneer_sdk2.Event.POINT_REACHED)

    threading.Thread(
        target=run_http_server,
        daemon=True,
    ).start()

    print(f"HTTP-сервер дрона запущен на порту {SERVER_PORT}")

    camera = Camera(camera_type=CameraType.MAIN)
    viewer = ImageViewer()
    servo_camera = ServoCamera()

    print(f"Установка камеры на угол {CAMERA_ANGLE}°")
    if servo_camera.set_angle(CAMERA_ANGLE) is False:
        raise RuntimeError("Не удалось установить угол камеры")

    print("Загрузка RKNN-модели:", MODEL_NAME)
    model = Yolo(
        model_name=MODEL_NAME,
        object_thresh=OBJECT_THRESHOLD,
        nms_thresh=NMS_THRESHOLD,
        img_width=INPUT_WIDTH,
        img_height=INPUT_HEIGHT,
    )

    print("RKNN-модель загружена")

    if PIXEL_TO_BODY_H is None:
        print(
            "ВНИМАНИЕ: гомография не настроена. "
            "Координата фиксируется только для панели около центра кадра."
        )
    else:
        print("Гомография камеры загружена")

    print("Ожидание команды POST /start")
    start_event.wait()

    with state_lock:
        mission_started = True
        start_event.clear()

    print("Команда принята. Начинается полёт")

    if drone.arm() is False:
        raise RuntimeError("Не удалось включить двигатели")

    time.sleep(3)

    if drone.takeoff() is False:
        raise RuntimeError("Не удалось выполнить взлёт")

    airborne = True
    print("Взлёт завершён")
    time.sleep(3)

    vision_stop_event.clear()
    vision_thread = threading.Thread(target=vision_loop, daemon=False)
    vision_thread.start()

    time.sleep(1)

    # Точка взлёта — первая известная позиция.
    scan_at_known_pose(0.0, 0.0, FLIGHT_HEIGHT)

    for point_number, point in enumerate(ROUTE, start=1):
        print(f"Точка маршрута №{point_number}")
        fly_to_point(
            drone_object=drone,
            x=point[0],
            y=point[1],
            z=point[2],
            yaw=point[3],
            duration=point[4],
            scan_after=True,
        )

    print("Возвращение к точке посадки")
    fly_to_point(
        drone_object=drone,
        x=LANDING_X,
        y=LANDING_Y,
        z=RETURN_HEIGHT,
        yaw=0.0,
        duration=0,
        scan_after=False,
    )

    time.sleep(3)

    print("Снижение над точкой посадки")
    fly_to_point(
        drone_object=drone,
        x=LANDING_X,
        y=LANDING_Y,
        z=PRELAND_HEIGHT,
        yaw=0.0,
        duration=0,
        scan_after=False,
    )

    time.sleep(3)

    print("Маршрут завершён. Выполняется посадка")
    if drone.land() is False:
        raise RuntimeError("Команда посадки была отклонена")

    airborne = False
    print("Посадка завершена")
    stop_vision()

except KeyboardInterrupt:
    print("Программа остановлена пользователем")

    if drone is not None and airborne:
        try:
            drone.land()
            airborne = False
        except Exception as landing_error:
            print("Ошибка аварийной посадки:", landing_error)

except Exception as error:
    print("Ошибка:", type(error).__name__, error)

    if drone is not None and airborne:
        try:
            drone.land()
            airborne = False
        except Exception as landing_error:
            print("Ошибка аварийной посадки:", landing_error)

finally:
    print("Освобождение ресурсов")
    stop_vision()

    if model is not None:
        try:
            model.release()
        except Exception as error:
            print("Ошибка освобождения RKNN:", error)

    if viewer is not None:
        try:
            viewer.close()
        except Exception as error:
            print("Ошибка закрытия трансляции:", error)

    if camera is not None:
        try:
            camera.stop()
        except Exception as error:
            print("Ошибка остановки камеры:", error)

    if drone is not None:
        try:
            drone.close_connection()
        except Exception as error:
            print("Ошибка закрытия Pioneer:", error)

    print("Ресурсы освобождены")
