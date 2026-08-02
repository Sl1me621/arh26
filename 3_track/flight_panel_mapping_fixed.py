from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np
import pioneer_sdk2
import requests

from flask import Flask, jsonify
from pioneer_rknn import Yolo
from pioneer_sdk2 import (
    Camera,
    CameraType,
    ImageViewer,
    Pioneer,
    ServoCamera,
)


# ============================================================
# СЕТЬ
# ============================================================

# Сервер на дроне: сюда компьютер отправляет POST /start.
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001

# Компьютер, на котором запущен актуальный map.py.
MAP_SERVER = "http://172.17.49.96:5001"
STATION_URL = f"{MAP_SERVER}/drone"
POSE_URL = f"{MAP_SERVER}/pose"
HEALTH_URL = f"{MAP_SERVER}/health"
CLEAR_URL = f"{MAP_SERVER}/clear"

HTTP_TIMEOUT = 1.5
HTTP_RETRIES = 3
CLEAR_MAP_ON_START = True



# ============================================================
# ПОЛЁТ
# ============================================================

FLIGHT_HEIGHT = 1.8

# x, y, z, yaw, time
ROUTE = [
    (0.3, 2.5, FLIGHT_HEIGHT, 0.0, 0),
    (0.3, 3.0, FLIGHT_HEIGHT, 0.0, 0),
    (-1.5, 3.0, FLIGHT_HEIGHT, 0.0, 0),
    (-1.5, 2.5, FLIGHT_HEIGHT, 0.0, 0),
    (-1.5, 2.0, FLIGHT_HEIGHT, 0.0, 0),
]

LANDING_X = 0
LANDING_Y = 0
RETURN_HEIGHT = 0.8
PRELAND_HEIGHT = 0.3
POINT_TIMEOUT = 30.0

# После взлёта и после каждой маршрутной точки дрон зависает,
# чтобы накопить несколько подтверждений панели.
SCAN_SECONDS = 2.5

# Геометрия поля используется для проверки маршрута до взлёта.
FIELD_WIDTH_M = 8.0
FIELD_HEIGHT_M = 6.0
DRONE_START_MAP_X = 1.5
DRONE_START_MAP_Y = 4.5
MAX_LEG_DISTANCE_M = 3.2


def validate_route() -> None:
    points = [(0.0, 0.0)] + [
        (float(point[0]), float(point[1]))
        for point in ROUTE
    ] + [(float(LANDING_X), float(LANDING_Y))]

    for x_local, y_local in points:
        # Локальный +X направлен вниз, локальный +Y — влево.
        map_x = DRONE_START_MAP_X + y_local
        map_y = DRONE_START_MAP_Y + x_local

        if not (
            0.0 <= map_x <= FIELD_WIDTH_M
            and 0.0 <= map_y <= FIELD_HEIGHT_M
        ):
            raise ValueError(
                f"Точка маршрута ({x_local:.2f}, {y_local:.2f}) "
                f"выходит за поле: map=({map_x:.2f}, {map_y:.2f})"
            )

    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        leg_distance = math.hypot(x2 - x1, y2 - y1)
        if leg_distance > MAX_LEG_DISTANCE_M:
            raise ValueError(
                f"Слишком длинный участок маршрута: "
                f"({x1:.2f}, {y1:.2f}) -> ({x2:.2f}, {y2:.2f}), "
                f"длина {leg_distance:.2f} м"
            )


# ============================================================
# КАМЕРА И RKNN
# ============================================================

CAMERA_ANGLE = -80
MODEL_NAME = "rknn_3576"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Общий порог должен быть не выше порога dust.
OBJECT_THRESHOLD = 0.55
NMS_THRESHOLD = 0.45

PANEL_CLASS_ID = 1
DUST_CLASS_ID = 0

PANEL_MIN_CONFIDENCE = 0.85
DUST_MIN_CONFIDENCE = 0.55

CLASS_NAMES = {
    DUST_CLASS_ID: "dust",
    PANEL_CLASS_ID: "panel",
}

CLASS_COLORS = {
    DUST_CLASS_ID: (0, 165, 255),
    PANEL_CLASS_ID: (0, 255, 0),
}

STATUS_HEALTHY = "исправна"
STATUS_DUSTY = "покрыта пылью"

STATUS_VIDEO_LABELS = {
    STATUS_HEALTHY: "OK",
    STATUS_DUSTY: "DUST",
}

STATUS_COLORS = {
    STATUS_HEALTHY: (0, 255, 0),
    STATUS_DUSTY: (0, 165, 255),
}

STREAM_NAME = "video"
STREAM_FPS = 15
DETECTION_PRINT_INTERVAL = 1.0



# ============================================================
# ВИЗУАЛЬНАЯ ОДОМЕТРИЯ
# ============================================================

# Стартовая локальная координата. На карте она соответствует
# точке (1.5, 4.5), преобразование выполняет map.py.
START_LOCAL_X = 0.0
START_LOCAL_Y = 0.0

# Калибровка по предыдущему полёту на высоте около 1.8 м.
# Чем точнее эти значения, тем точнее координаты и перемещение.
# Для кадра 640x640:
PIXELS_PER_METER_X = 280.0
PIXELS_PER_METER_Y = 450.0

# Поворот осей камеры относительно локальных осей поля.
# 0 означает текущую установленную ориентацию камеры.
CAMERA_TO_LOCAL_ROTATION_DEG = 0.0

# Локальная система:
#   +X направлен вниз по карте;
#   +Y направлен вправо по карте, поэтому при движении влево Y уменьшается.
#
# При текущей ориентации камеры:
#   вертикальный сдвиг изображения используется для локального X;
#   горизонтальный сдвиг изображения используется для локального Y.
# Знаки ниже можно поменять отдельно, если после теста движение
# отображается в противоположную сторону.
ODOMETRY_X_SIGN = -1.0
ODOMETRY_Y_SIGN = 1.0

# Положение панели относительно камеры:
#   объект ниже центра кадра имеет больший локальный X;
#   локальный +Y направлен влево, поэтому объект правее
#   центра кадра имеет меньший локальный Y.
OBJECT_X_SIGN = 1.0
OBJECT_Y_SIGN = -1.0

ODOMETRY_MAX_CORNERS = 500
ODOMETRY_QUALITY_LEVEL = 0.01
ODOMETRY_MIN_DISTANCE_PX = 8
ODOMETRY_MIN_FEATURES = 25
ODOMETRY_RANSAC_THRESHOLD_PX = 2.5
ODOMETRY_MAX_FLOW_PX = 100.0
ODOMETRY_MAX_ROTATION_DEG = 4.0
ODOMETRY_MIN_SCALE = 0.94
ODOMETRY_MAX_SCALE = 1.06

POSE_SEND_INTERVAL = 0.25
POSE_PATH_MIN_STEP_M = 0.03



# ============================================================
# ОПРЕДЕЛЕНИЕ ПЫЛИ НА ПАНЕЛИ
# ============================================================

PANEL_MARGIN_PX = 10
DUST_OVERLAP_THRESHOLD = 0.15

MIN_DUST_VOTES = 2
DUST_VOTE_RATIO = 0.25



# ============================================================
# ОБЪЕДИНЕНИЕ ПОВТОРНЫХ НАБЛЮДЕНИЙ
# ============================================================

# Наблюдения одной панели внутри этого радиуса объединяются.
STATION_MERGE_RADIUS_M = 0.8
MIN_STATION_CONFIRMATIONS = 8
STATION_UPDATE_INTERVAL = 0.75
STATION_POSITION_UPDATE_M = 0.2
MAX_STATIONS = 5

stations_lock = threading.Lock()
station_tracks: list[dict] = []
next_station_id = 1



# ============================================================
# СОСТОЯНИЕ
# ============================================================

app = Flask(__name__)

point_event = threading.Event()
start_event = threading.Event()
vision_stop_event = threading.Event()

# Регистрация станций разрешена только во время зависания
# в заранее известной маршрутной точке.
measurement_enabled = threading.Event()

state_lock = threading.Lock()
stable_pose_lock = threading.Lock()

# Точная командная координата достигнутой маршрутной точки.
# Она используется как база для расчёта координат панели.
stable_drone_pose: tuple[float, float, float] | None = None

route_distance_m = 0.0
last_waypoint_pose: tuple[float, float] | None = None

mission_started = False
vision_running = False
airborne = False

processed_frames = 0
last_inference_ms = 0.0
last_detections: list[dict] = []

drone: Pioneer | None = None
camera: Camera | None = None
viewer: ImageViewer | None = None
servo_camera: ServoCamera | None = None
model: Yolo | None = None
vision_thread: threading.Thread | None = None



# ============================================================
# ВСПОМОГАТЕЛЬНАЯ МАТЕМАТИКА
# ============================================================

def rotate_vector(
    x: float,
    y: float,
    angle_deg: float,
) -> tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    return (
        x * cos_a - y * sin_a,
        x * sin_a + y * cos_a,
    )



# ============================================================
# КЛАСС ВИЗУАЛЬНОЙ ОДОМЕТРИИ
# ============================================================

class VisualOdometry:
    def __init__(self) -> None:
        self.x = float(START_LOCAL_X)
        self.y = float(START_LOCAL_Y)

        self.previous_gray: np.ndarray | None = None

        self.valid = False
        self.tracked_features = 0
        self.inliers = 0
        self.flow_u = 0.0
        self.flow_v = 0.0
        self.delta_x = 0.0
        self.delta_y = 0.0
        self.rotation_deg = 0.0
        self.scale = 1.0
        self.total_distance = 0.0

        self._lock = threading.Lock()
        self._epoch = 0

    def reset(self) -> None:
        with self._lock:
            self.x = float(START_LOCAL_X)
            self.y = float(START_LOCAL_Y)
            self.previous_gray = None
            self.valid = False
            self.tracked_features = 0
            self.inliers = 0
            self.flow_u = 0.0
            self.flow_v = 0.0
            self.delta_x = 0.0
            self.delta_y = 0.0
            self.rotation_deg = 0.0
            self.scale = 1.0
            self.total_distance = 0.0
            self._epoch += 1

    def set_pose(
        self,
        x: float,
        y: float,
        reset_frame: bool = True,
    ) -> None:
        """Привязывает визуальную одометрию к известной точке маршрута."""

        with self._lock:
            self.x = float(x)
            self.y = float(y)

            if reset_frame:
                self.previous_gray = None

            self.valid = True
            self.delta_x = 0.0
            self.delta_y = 0.0
            self.flow_u = 0.0
            self.flow_v = 0.0
            self._epoch += 1

    def get_pose(self) -> tuple[float, float]:
        with self._lock:
            return float(self.x), float(self.y)

    def get_state(self) -> dict:
        with self._lock:
            return {
                "x": float(self.x),
                "y": float(self.y),
                "valid": bool(self.valid),
                "tracked_features": int(self.tracked_features),
                "inliers": int(self.inliers),
                "flow_u": float(self.flow_u),
                "flow_v": float(self.flow_v),
                "delta_x": float(self.delta_x),
                "delta_y": float(self.delta_y),
                "rotation_deg": float(self.rotation_deg),
                "scale": float(self.scale),
                "total_distance": float(self.total_distance),
                "coordinate_mode": "visual_odometry",
            }

    def update(self, frame_bgr: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        with self._lock:
            previous_gray = self.previous_gray
            update_epoch = self._epoch
            self.previous_gray = gray

        if previous_gray is None:
            with self._lock:
                self.valid = False
            return self.get_state()

        previous_points = cv2.goodFeaturesToTrack(
            previous_gray,
            maxCorners=ODOMETRY_MAX_CORNERS,
            qualityLevel=ODOMETRY_QUALITY_LEVEL,
            minDistance=ODOMETRY_MIN_DISTANCE_PX,
            blockSize=7,
        )

        if previous_points is None or len(previous_points) < ODOMETRY_MIN_FEATURES:
            with self._lock:
                self.valid = False
                self.tracked_features = 0 if previous_points is None else len(previous_points)
                self.inliers = 0
            return self.get_state()

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if current_points is None or status is None:
            with self._lock:
                self.valid = False
                self.tracked_features = 0
                self.inliers = 0
            return self.get_state()

        good_mask = status.reshape(-1) == 1
        previous_good = previous_points.reshape(-1, 2)[good_mask]
        current_good = current_points.reshape(-1, 2)[good_mask]

        tracked_count = len(previous_good)

        if tracked_count < ODOMETRY_MIN_FEATURES:
            with self._lock:
                self.valid = False
                self.tracked_features = tracked_count
                self.inliers = 0
            return self.get_state()

        affine, inlier_mask = cv2.estimateAffinePartial2D(
            previous_good,
            current_good,
            method=cv2.RANSAC,
            ransacReprojThreshold=ODOMETRY_RANSAC_THRESHOLD_PX,
            maxIters=2000,
            confidence=0.99,
            refineIters=10,
        )

        if affine is None:
            with self._lock:
                self.valid = False
                self.tracked_features = tracked_count
                self.inliers = 0
            return self.get_state()

        inliers = (
            int(np.count_nonzero(inlier_mask))
            if inlier_mask is not None
            else tracked_count
        )

        if inliers < ODOMETRY_MIN_FEATURES:
            with self._lock:
                self.valid = False
                self.tracked_features = tracked_count
                self.inliers = inliers
            return self.get_state()

        a00, a01, tx = [float(value) for value in affine[0]]
        a10, a11, ty = [float(value) for value in affine[1]]

        scale = math.sqrt(a00 * a00 + a10 * a10)
        rotation_deg = math.degrees(math.atan2(a10, a00))

        center = np.array(
            [INPUT_WIDTH / 2.0, INPUT_HEIGHT / 2.0, 1.0],
            dtype=np.float64,
        )
        transformed_center = affine @ center

        flow_u = float(transformed_center[0] - center[0])
        flow_v = float(transformed_center[1] - center[1])
        flow_length = float(math.hypot(flow_u, flow_v))

        transform_is_valid = (
            ODOMETRY_MIN_SCALE <= scale <= ODOMETRY_MAX_SCALE
            and abs(rotation_deg) <= ODOMETRY_MAX_ROTATION_DEG
            and flow_length <= ODOMETRY_MAX_FLOW_PX
        )

        if not transform_is_valid:
            with self._lock:
                self.valid = False
                self.tracked_features = tracked_count
                self.inliers = inliers
                self.flow_u = flow_u
                self.flow_v = flow_v
                self.rotation_deg = rotation_deg
                self.scale = scale
            return self.get_state()

        # Локальный X направлен вниз, поэтому берём вертикальную
        # составляющую optical flow. Локальный Y направлен влево,
        # поэтому берём горизонтальную составляющую.
        camera_dx = (
            ODOMETRY_X_SIGN * flow_v / PIXELS_PER_METER_Y
        )
        camera_dy = (
            ODOMETRY_Y_SIGN * flow_u / PIXELS_PER_METER_X
        )

        camera_dx, camera_dy = rotate_vector(
            camera_dx,
            camera_dy,
            CAMERA_TO_LOCAL_ROTATION_DEG,
        )

        stale_update = False

        with self._lock:
            # Если основной поток за время расчёта привязал позу
            # к маршрутной точке, старый optical-flow сдвиг отбрасываем.
            if update_epoch != self._epoch:
                stale_update = True
            else:
                self.x += camera_dx
                self.y += camera_dy
                self.valid = True
                self.tracked_features = tracked_count
                self.inliers = inliers
                self.flow_u = flow_u
                self.flow_v = flow_v
                self.delta_x = camera_dx
                self.delta_y = camera_dy
                self.rotation_deg = rotation_deg
                self.scale = scale
                self.total_distance += math.hypot(
                    camera_dx,
                    camera_dy,
                )

        if stale_update:
            return self.get_state()

        return self.get_state()


visual_odometry = VisualOdometry()



# ============================================================
# ЦЕНТР ПАНЕЛИ -> ЛОКАЛЬНЫЕ КООРДИНАТЫ
# ============================================================

def estimate_station_coordinates(
    center_x: float,
    center_y: float,
) -> tuple[float, float, str] | None:
    """
    Возвращает координаты станции только во время сканирования
    в достигнутой маршрутной точке.

    Базовая координата берётся не из визуальной одометрии,
    а из точной команды маршрута. Центр бокса добавляет только
    смещение панели относительно центра кадра.
    """

    if not measurement_enabled.is_set():
        return None

    with stable_pose_lock:
        pose = stable_drone_pose

    if pose is None:
        return None

    drone_x, drone_y, _ = pose

    # Локальный X соответствует вертикальному направлению кадра.
    offset_x = (
        OBJECT_X_SIGN
        * (center_y - INPUT_HEIGHT / 2.0)
        / PIXELS_PER_METER_Y
    )

    # Локальный Y соответствует горизонтальному направлению кадра.
    offset_y = (
        OBJECT_Y_SIGN
        * (center_x - INPUT_WIDTH / 2.0)
        / PIXELS_PER_METER_X
    )

    offset_x, offset_y = rotate_vector(
        offset_x,
        offset_y,
        CAMERA_TO_LOCAL_ROTATION_DEG,
    )

    return (
        drone_x + offset_x,
        drone_y + offset_y,
        "waypoint_offset",
    )



# ============================================================
# ГЕОМЕТРИЯ БОКСОВ
# ============================================================

def clamp_box(
    box,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
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


def expand_box(
    box: tuple[int, int, int, int],
    margin: int,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width - 1, x2 + margin),
        min(height - 1, y2 + margin),
    )


def box_center(
    box: tuple[int, int, int, int],
) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def dust_coverage_inside_panel(
    panel_box: tuple[int, int, int, int],
    dust_box: tuple[int, int, int, int],
) -> float:
    panel_x1, panel_y1, panel_x2, panel_y2 = panel_box
    dust_x1, dust_y1, dust_x2, dust_y2 = dust_box

    intersection_x1 = max(panel_x1, dust_x1)
    intersection_y1 = max(panel_y1, dust_y1)
    intersection_x2 = min(panel_x2, dust_x2)
    intersection_y2 = min(panel_y2, dust_y2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection_area = intersection_width * intersection_height

    dust_area = max(
        1,
        (dust_x2 - dust_x1) * (dust_y2 - dust_y1),
    )

    return float(intersection_area / dust_area)


def assign_dust_to_panels(
    panels: list[dict],
    dust_objects: list[dict],
    width: int,
    height: int,
) -> dict[int, list[dict]]:
    matches: dict[int, list[dict]] = {
        panel_index: []
        for panel_index in range(len(panels))
    }

    for dust in dust_objects:
        best_panel_index: int | None = None
        best_coverage = 0.0

        for panel_index, panel in enumerate(panels):
            expanded_panel = expand_box(
                panel["box"],
                PANEL_MARGIN_PX,
                width,
                height,
            )

            coverage = dust_coverage_inside_panel(
                expanded_panel,
                dust["box"],
            )

            if coverage > best_coverage:
                best_coverage = coverage
                best_panel_index = panel_index

        if (
            best_panel_index is not None
            and best_coverage >= DUST_OVERLAP_THRESHOLD
        ):
            matches[best_panel_index].append({
                "dust": dust,
                "coverage": best_coverage,
            })

    return matches



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
    pose_state = visual_odometry.get_state()

    with state_lock:
        response = {
            "mission_started": mission_started,
            "vision_running": vision_running,
            "processed_frames": processed_frames,
            "inference_ms": round(last_inference_ms, 2),
            "detections": list(last_detections),
            "pose": {
                "x": round(pose_state["x"], 3),
                "y": round(pose_state["y"], 3),
                "valid": pose_state["valid"],
                "inliers": pose_state["inliers"],
                "coordinate_mode": pose_state["coordinate_mode"],
            },
            "measurement_enabled": measurement_enabled.is_set(),
            "stream": (
                f"http://172.17.49.101:8889/{STREAM_NAME}/"
            ),
        }

    with stations_lock:
        response["stations"] = [
            {
                "id": int(track["id"]),
                "x": round(float(track["x"]), 3),
                "y": round(float(track["y"]), 3),
                "status": str(track["status"]),
                "confirmations": int(track["count"]),
                "dust_votes": int(track["dust_votes"]),
                "clean_votes": int(track["clean_votes"]),
                "sent": bool(track["sent"]),
            }
            for track in station_tracks
        ]

    return jsonify(response), 200


def run_http_server() -> None:
    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )



# ============================================================
# HTTP-КЛИЕНТ КАРТЫ
# ============================================================

def check_map_server() -> bool:
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        print("[HTTP] map.py доступен:", response.text)
        return True
    except requests.RequestException as error:
        print("[HTTP] map.py недоступен:", error)
        return False


def clear_map() -> None:
    try:
        response = requests.post(
            CLEAR_URL,
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        print("[HTTP] Карта очищена")
    except requests.RequestException as error:
        print("[HTTP] Не удалось очистить карту:", error)


def send_json_with_retries(
    url: str,
    payload: dict,
    description: str,
) -> bool:
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=HTTP_TIMEOUT,
            )
            response.raise_for_status()
            return True
        except requests.RequestException as error:
            if attempt == HTTP_RETRIES:
                print(f"[HTTP] {description}: {error}")
            else:
                time.sleep(0.25)

    return False


def send_pose_if_needed(
    pose_state: dict,
    force: bool = False,
) -> None:
    now = time.monotonic()

    previous_time = getattr(
        send_pose_if_needed,
        "last_send_time",
        0.0,
    )
    previous_x = getattr(
        send_pose_if_needed,
        "last_x",
        float("nan"),
    )
    previous_y = getattr(
        send_pose_if_needed,
        "last_y",
        float("nan"),
    )

    moved = (
        not math.isfinite(previous_x)
        or math.hypot(
            pose_state["x"] - previous_x,
            pose_state["y"] - previous_y,
        ) >= POSE_PATH_MIN_STEP_M
    )

    if not force and (
        now - previous_time < POSE_SEND_INTERVAL
        or not moved
    ):
        return

    payload = {
        "x": round(float(pose_state["x"]), 4),
        "y": round(float(pose_state["y"]), 4),
        "valid": bool(pose_state["valid"]),
        "tracked_features": int(pose_state["tracked_features"]),
        "inliers": int(pose_state["inliers"]),
        "flow_u": round(float(pose_state["flow_u"]), 3),
        "flow_v": round(float(pose_state["flow_v"]), 3),
        "rotation_deg": round(float(pose_state["rotation_deg"]), 3),
        "total_distance": round(float(pose_state["total_distance"]), 3),
        "coordinate_mode": "visual_odometry",
    }

    if send_json_with_retries(
        POSE_URL,
        payload,
        "не удалось отправить позу",
    ):
        send_pose_if_needed.last_send_time = now
        send_pose_if_needed.last_x = pose_state["x"]
        send_pose_if_needed.last_y = pose_state["y"]


def send_station_payload(track_id: int, payload: dict) -> None:
    success = send_json_with_retries(
        STATION_URL,
        payload,
        f"станция №{track_id} не отправлена",
    )

    if success:
        print(
            f"[HTTP] Станция №{track_id}: "
            f"x={payload['x']:.3f}, "
            f"y={payload['y']:.3f}, "
            f"status={payload['status']}, "
            f"n={payload['confirmations']}"
        )

    with stations_lock:
        for track in station_tracks:
            if track["id"] != track_id:
                continue

            if success:
                track["sent"] = True
                track["last_sent_status"] = payload["status"]
                track["last_sent_x"] = payload["x"]
                track["last_sent_y"] = payload["y"]
                track["last_send_time"] = time.monotonic()

            track["sending"] = False
            break



# ============================================================
# ПОЛЁТ И КОРРЕКЦИЯ ПОЗЫ
# ============================================================

def point_reached(event) -> None:
    point_event.set()


def wait_for_point(timeout: float = POINT_TIMEOUT) -> None:
    if not point_event.wait(timeout=timeout):
        raise TimeoutError(
            f"Дрон не достиг точки за {timeout} секунд"
        )

    point_event.clear()


def anchor_pose_to_waypoint(
    x: float,
    y: float,
    description: str,
) -> None:
    """Привязывает координаты к реально достигнутой точке маршрута."""
    global route_distance_m
    global last_waypoint_pose

    visual_odometry.set_pose(
        x=x,
        y=y,
        reset_frame=True,
    )

    with state_lock:
        if last_waypoint_pose is not None:
            route_distance_m += math.hypot(
                x - last_waypoint_pose[0],
                y - last_waypoint_pose[1],
            )

        last_waypoint_pose = (float(x), float(y))
        current_route_distance = float(route_distance_m)

    waypoint_pose = {
        "x": float(x),
        "y": float(y),
        "valid": True,
        "tracked_features": 0,
        "inliers": 0,
        "flow_u": 0.0,
        "flow_v": 0.0,
        "rotation_deg": 0.0,
        "total_distance": current_route_distance,
        "coordinate_mode": "waypoint",
    }

    send_pose_if_needed(
        waypoint_pose,
        force=True,
    )

    print(
        f"[ПОЗА] {description}: "
        f"x={x:.2f}, y={y:.2f}"
    )


def scan_at_known_pose(
    x: float,
    y: float,
    z: float,
) -> None:
    global stable_drone_pose

    anchor_pose_to_waypoint(
        x=x,
        y=y,
        description="привязка к известной точке",
    )

    with stable_pose_lock:
        stable_drone_pose = (
            float(x),
            float(y),
            float(z),
        )

    measurement_enabled.set()

    print(
        f"Сканирование включено: x={x:.2f}, y={y:.2f}, "
        f"z={z:.2f}, {SCAN_SECONDS:.1f} с"
    )

    try:
        time.sleep(SCAN_SECONDS)
    finally:
        measurement_enabled.clear()
        print("Сканирование выключено")


def fly_to_point(
    drone_object: Pioneer,
    x: float,
    y: float,
    z: float,
    yaw: float,
    duration: int,
    scan_after: bool = True,
) -> None:
    # Во время движения детекции видны в видео,
    # но станции не регистрируются и не отправляются на карту.
    measurement_enabled.clear()
    point_event.clear()

    print(
        f"Летим в точку: x={x}, y={y}, z={z}, "
        f"yaw={yaw}, time={duration}"
    )

    command_sent = drone_object.go_to_local_point(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        time=duration,
    )

    if command_sent is False:
        raise RuntimeError(
            f"Команда перехода в точку "
            f"x={x}, y={y}, z={z} отклонена"
        )

    wait_for_point()
    print("Точка достигнута")

    # После POINT_REACHED командная координата считается точной
    # и используется для коррекции визуальной одометрии.
    if scan_after:
        scan_at_known_pose(x, y, z)
    else:
        anchor_pose_to_waypoint(
            x=x,
            y=y,
            description="точка достигнута",
        )



# ============================================================
# ТРЕКИ СТАНЦИЙ
# ============================================================

def find_matching_track(x: float, y: float) -> dict | None:
    best_track = None
    best_distance = float("inf")

    for track in station_tracks:
        distance = float(math.hypot(
            track["x"] - x,
            track["y"] - y,
        ))

        if (
            distance <= STATION_MERGE_RADIUS_M
            and distance < best_distance
        ):
            best_track = track
            best_distance = distance

    return best_track


def register_station_observation(
    x: float,
    y: float,
    frame_status: str,
    confidence: float,
) -> None:
    global next_station_id

    payload_to_send: tuple[int, dict] | None = None
    frame_is_dusty = frame_status == STATUS_DUSTY

    with stations_lock:
        track = find_matching_track(x, y)

        if track is None:
            if len(station_tracks) >= MAX_STATIONS:
                return

            track = {
                "id": next_station_id,
                "x": float(x),
                "y": float(y),
                "count": 1,
                "confidence_sum": float(confidence),
                "dust_votes": 1 if frame_is_dusty else 0,
                "clean_votes": 0 if frame_is_dusty else 1,
                "dust_confirmed": False,
                "status": STATUS_HEALTHY,
                "sent": False,
                "sending": False,
                "last_sent_status": None,
                "last_sent_x": None,
                "last_sent_y": None,
                "last_send_time": 0.0,
            }
            station_tracks.append(track)
            next_station_id += 1

            print(
                f"Новый кандидат станции №{track['id']}: "
                f"local=({x:.2f}, {y:.2f}), "
                f"status={frame_status}"
            )
        else:
            old_count = track["count"]
            new_count = old_count + 1

            track["x"] = (
                track["x"] * old_count + x
            ) / new_count
            track["y"] = (
                track["y"] * old_count + y
            ) / new_count
            track["count"] = new_count
            track["confidence_sum"] += confidence

            if frame_is_dusty:
                track["dust_votes"] += 1
            else:
                track["clean_votes"] += 1

        total_votes = track["dust_votes"] + track["clean_votes"]
        dust_ratio = track["dust_votes"] / max(1, total_votes)

        if (
            track["dust_votes"] >= MIN_DUST_VOTES
            and dust_ratio >= DUST_VOTE_RATIO
        ):
            track["dust_confirmed"] = True

        resolved_status = (
            STATUS_DUSTY
            if track["dust_confirmed"]
            else STATUS_HEALTHY
        )
        track["status"] = resolved_status

        enough_confirmations = (
            track["count"] >= MIN_STATION_CONFIRMATIONS
        )
        now = time.monotonic()

        never_sent = not track["sent"]
        status_changed = (
            track["sent"]
            and track["last_sent_status"] != resolved_status
        )

        position_changed = False
        if track["sent"]:
            position_changed = math.hypot(
                track["x"] - float(track["last_sent_x"]),
                track["y"] - float(track["last_sent_y"]),
            ) >= STATION_POSITION_UPDATE_M

        periodic_update = (
            track["sent"]
            and now - track["last_send_time"]
            >= STATION_UPDATE_INTERVAL
        )

        should_send = (
            enough_confirmations
            and not track["sending"]
            and (
                never_sent
                or status_changed
                or position_changed
                or periodic_update
            )
        )

        if should_send:
            track["sending"] = True

            payload = {
                "id": int(track["id"]),
                "x": round(float(track["x"]), 3),
                "y": round(float(track["y"]), 3),
                "status": resolved_status,
                "confidence": round(
                    track["confidence_sum"] / track["count"],
                    3,
                ),
                "confirmations": int(track["count"]),
                "dust_votes": int(track["dust_votes"]),
                "clean_votes": int(track["clean_votes"]),
                "coordinate_mode": "visual_odometry",
            }

            payload_to_send = int(track["id"]), payload

    if payload_to_send is not None:
        threading.Thread(
            target=send_station_payload,
            args=payload_to_send,
            daemon=True,
        ).start()



# ============================================================
# YOLO: PANEL + DUST
# ============================================================

def parse_yolo_objects(
    result,
    width: int,
    height: int,
) -> list[dict]:
    if result is None:
        return []

    if not isinstance(result, (tuple, list)) or len(result) != 3:
        print("Неожиданный результат YOLO:", type(result))
        return []

    boxes, classes, scores = result

    if boxes is None or classes is None or scores is None:
        return []

    objects: list[dict] = []

    for box, class_id, score in zip(boxes, classes, scores):
        class_id = int(class_id)
        confidence = float(score)

        if class_id == PANEL_CLASS_ID:
            if confidence < PANEL_MIN_CONFIDENCE:
                continue
        elif class_id == DUST_CLASS_ID:
            if confidence < DUST_MIN_CONFIDENCE:
                continue
        else:
            continue

        normalized_box = clamp_box(box, width, height)

        if normalized_box is None:
            continue

        objects.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "confidence": confidence,
            "box": normalized_box,
            "center": box_center(normalized_box),
        })

    return objects


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    x = max(2, int(x))
    y = max(20, int(y))

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        1,
        cv2.LINE_AA,
    )


def draw_and_register_detections(
    frame: np.ndarray,
    result,
    last_print_times: dict[str, float],
) -> int:
    height, width = frame.shape[:2]

    objects = parse_yolo_objects(result, width, height)

    panels = [
        obj for obj in objects
        if obj["class_id"] == PANEL_CLASS_ID
    ]
    dust_objects = [
        obj for obj in objects
        if obj["class_id"] == DUST_CLASS_ID
    ]

    matches = assign_dust_to_panels(
        panels,
        dust_objects,
        width,
        height,
    )

    for dust in dust_objects:
        x1, y1, x2, y2 = dust["box"]
        color = CLASS_COLORS[DUST_CLASS_ID]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(
            frame,
            f"dust {dust['confidence']:.2f}",
            x1,
            y1,
            color,
        )

    for panel_index, panel in enumerate(panels):
        x1, y1, x2, y2 = panel["box"]
        center_x, center_y = panel["center"]

        matched_dust = matches.get(panel_index, [])
        frame_status = (
            STATUS_DUSTY
            if matched_dust
            else STATUS_HEALTHY
        )

        color = STATUS_COLORS[frame_status]
        video_status = STATUS_VIDEO_LABELS[frame_status]

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        draw_label(
            frame,
            f"panel {video_status} {panel['confidence']:.2f}",
            x1,
            y1,
            color,
        )

        station_coordinates = estimate_station_coordinates(
            center_x,
            center_y,
        )

        cv2.drawMarker(
            frame,
            (int(round(center_x)), int(round(center_y))),
            (255, 255, 255),
            cv2.MARKER_CROSS,
            18,
            2,
            cv2.LINE_AA,
        )

        if station_coordinates is not None:
            local_x, local_y, coordinate_mode = station_coordinates

            coordinate_text = (
                f"local=({local_x:.2f},{local_y:.2f}) "
                f"{coordinate_mode}"
            )
            coordinate_color = (255, 255, 255)

            register_station_observation(
                x=local_x,
                y=local_y,
                frame_status=frame_status,
                confidence=panel["confidence"],
            )
        else:
            # Во время перелёта объект только отображается в видео.
            # В station_tracks он не попадает.
            local_x = None
            local_y = None
            coordinate_text = "SCAN OFF"
            coordinate_color = (0, 0, 255)

        draw_label(
            frame,
            coordinate_text,
            int(center_x + 10),
            int(center_y - 10),
            coordinate_color,
        )

        for match in matched_dust:
            dust_center_x, dust_center_y = match["dust"]["center"]
            cv2.line(
                frame,
                (int(center_x), int(center_y)),
                (int(dust_center_x), int(dust_center_y)),
                color,
                2,
                cv2.LINE_AA,
            )

        print_key = f"panel_{video_status}"
        now = time.monotonic()

        if (
            now - last_print_times.get(print_key, 0.0)
            >= DETECTION_PRINT_INTERVAL
        ):
            if station_coordinates is not None:
                position_text = (
                    f"local=({local_x:.2f}, {local_y:.2f})"
                )
            else:
                position_text = "SCAN OFF"

            print(
                f"Панель {video_status}: "
                f"{position_text}, "
                f"confidence={panel['confidence']:.3f}, "
                f"dust={len(matched_dust)}"
            )
            last_print_times[print_key] = now

    return len(objects)


def draw_statistics(
    frame: np.ndarray,
    inference_ms: float,
    objects_count: int,
    pose_state: dict,
) -> None:
    lines = [
        f"Inference: {inference_ms:.1f} ms",
        f"Objects: {objects_count}",
        (
            f"Pose: ({pose_state['x']:.2f},"
            f" {pose_state['y']:.2f})"
        ),
        (
            f"Odom: {'OK' if pose_state['valid'] else 'WAIT'} "
            f"inliers={pose_state['inliers']}"
        ),
        (
            f"Flow: ({pose_state['flow_u']:.1f},"
            f" {pose_state['flow_v']:.1f}) px"
        ),
        (
            "Registration: ON"
            if measurement_enabled.is_set()
            else "Registration: OFF"
        ),
    ]

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (8, 8),
        (345, 170),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 24



# ============================================================
# ПОТОК КАМЕРЫ И NPU
# ============================================================

def vision_loop() -> None:
    global vision_running
    global processed_frames
    global last_inference_ms
    global last_detections

    if camera is None or viewer is None or model is None:
        raise RuntimeError(
            "Камера, ImageViewer или модель не инициализированы"
        )

    frame_number = 0
    last_print_times: dict[str, float] = {}

    with state_lock:
        vision_running = True

    print("Обработка камеры, RKNN и одометрии запущена")
    print(
        "Трансляция:",
        f"http://172.17.49.101:8889/{STREAM_NAME}/",
    )

    # Путь на карту отправляется только после достижения waypoint.

    try:
        while not vision_stop_event.is_set():
            frame = camera.get_cv_frame(timeout=1.0)

            if frame is None:
                print("Камера не вернула кадр")
                continue

            display_frame = cv2.resize(
                frame,
                (INPUT_WIDTH, INPUT_HEIGHT),
                interpolation=cv2.INTER_LINEAR,
            )

            # Визуальная одометрия работает во время всего полёта.
            # После достижения каждой маршрутной точки её координата
            # корректируется командной координатой в основном потоке.
            pose_state = visual_odometry.update(
                display_frame
            )

            # Визуальная одометрия остаётся только диагностикой.
            # На карту во время перелёта она не отправляется,
            # потому что может накапливать ошибку в несколько метров.

            rgb = cv2.cvtColor(
                display_frame,
                cv2.COLOR_BGR2RGB,
            )

            input_tensor = np.expand_dims(
                rgb,
                axis=0,
            )

            input_tensor = np.ascontiguousarray(
                input_tensor,
                dtype=np.uint8,
            )

            inference_start = time.perf_counter()

            result = model.run([input_tensor])

            inference_ms = (
                time.perf_counter() - inference_start
            ) * 1000.0

            objects_count = draw_and_register_detections(
                display_frame,
                result,
                last_print_times,
            )

            draw_statistics(
                display_frame,
                inference_ms,
                objects_count,
                pose_state,
            )

            viewer.imshow(
                name=STREAM_NAME,
                frame=display_frame,
                fps=STREAM_FPS,
            )

            frame_number += 1

            with state_lock:
                processed_frames = frame_number
                last_inference_ms = inference_ms

                # Возвращаемый draw_and_register_detections —
                # количество объектов. Для /status отдельно
                # сохраняем текущие треки станций ниже.
                last_detections = []

            with stations_lock:
                station_snapshot = [
                    {
                        "id": int(track["id"]),
                        "x": round(float(track["x"]), 3),
                        "y": round(float(track["y"]), 3),
                        "status": str(track["status"]),
                        "confirmations": int(track["count"]),
                    }
                    for track in station_tracks
                ]

            with state_lock:
                last_detections = station_snapshot

            if frame_number % 30 == 0:
                print(
                    f"Кадр {frame_number} | "
                    f"pose=({pose_state['x']:.2f},"
                    f" {pose_state['y']:.2f}) | "
                    f"odom="
                    f"{'OK' if pose_state['valid'] else 'WAIT'} | "
                    f"inference={inference_ms:.1f} мс | "
                    f"объектов={objects_count}"
                )

    except Exception as error:
        print(
            "Ошибка в потоке камеры:",
            type(error).__name__,
            error,
        )

    finally:
        with state_lock:
            vision_running = False

        print("Обработка камеры и RKNN остановлена")


def stop_vision() -> None:
    global vision_thread

    measurement_enabled.clear()
    vision_stop_event.set()

    if (
        vision_thread is not None
        and vision_thread.is_alive()
    ):
        vision_thread.join(timeout=3.0)



# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

try:
    validate_route()

    print("Подключение к автопилоту")

    drone = Pioneer()

    drone.subscribe(
        point_reached,
        pioneer_sdk2.Event.POINT_REACHED,
    )

    threading.Thread(
        target=run_http_server,
        daemon=True,
    ).start()

    print(
        f"HTTP-сервер дрона запущен на порту "
        f"{SERVER_PORT}"
    )

    print("Проверка map.py")

    if not check_map_server():
        raise RuntimeError(
            f"Карта недоступна по адресу {HEALTH_URL}. "
            "Полёт отменён. Проверь IP компьютера, map.py и брандмауэр."
        )

    if CLEAR_MAP_ON_START:
        clear_map()

    print("Запуск камеры")

    camera = Camera(
        camera_type=CameraType.MAIN
    )

    viewer = ImageViewer()
    servo_camera = ServoCamera()

    print(
        f"Установка камеры на угол "
        f"{CAMERA_ANGLE}°"
    )

    if servo_camera.set_angle(CAMERA_ANGLE) is False:
        raise RuntimeError(
            "Не удалось установить угол камеры"
        )

    print("Загрузка RKNN-модели:", MODEL_NAME)

    model = Yolo(
        model_name=MODEL_NAME,
        object_thresh=OBJECT_THRESHOLD,
        nms_thresh=NMS_THRESHOLD,
        img_width=INPUT_WIDTH,
        img_height=INPUT_HEIGHT,
    )

    print("RKNN-модель загружена")
    print("Ожидание команды POST /start")

    start_event.wait()

    with state_lock:
        mission_started = True
        start_event.clear()

    print("Команда принята. Начинается полёт")

    if drone.arm() is False:
        raise RuntimeError(
            "Не удалось включить двигатели"
        )

    time.sleep(3)

    if drone.takeoff() is False:
        raise RuntimeError(
            "Не удалось выполнить взлёт"
        )

    airborne = True
    print("Взлёт завершён")

    time.sleep(3)

    # После взлёта начинаем одометрию с локального (0, 0).
    visual_odometry.set_pose(
        START_LOCAL_X,
        START_LOCAL_Y,
        reset_frame=True,
    )

    vision_stop_event.clear()

    vision_thread = threading.Thread(
        target=vision_loop,
        daemon=False,
    )

    vision_thread.start()

    time.sleep(1)

    scan_at_known_pose(
        0.0,
        0.0,
        FLIGHT_HEIGHT,
    )

    for point_number, point in enumerate(
        ROUTE,
        start=1,
    ):
        print(
            f"Точка маршрута №{point_number}"
        )

        fly_to_point(
            drone_object=drone,
            x=point[0],
            y=point[1],
            z=point[2],
            yaw=point[3],
            duration=point[4],
            scan_after=True,
        )

    measurement_enabled.clear()
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

    print(
        "Маршрут завершён. "
        "Выполняется посадка"
    )

    if drone.land() is False:
        raise RuntimeError(
            "Команда посадки была отклонена"
        )

    airborne = False
    print("Посадка завершена")

    stop_vision()

except KeyboardInterrupt:
    print("Программа остановлена пользователем")

    if drone is not None and airborne:
        print("Выполняется аварийная посадка")

        try:
            drone.land()
            airborne = False
        except Exception as landing_error:
            print(
                "Ошибка аварийной посадки:",
                landing_error,
            )

except Exception as error:
    print(
        "Ошибка:",
        type(error).__name__,
        error,
    )

    if drone is not None and airborne:
        print("Выполняется посадка из-за ошибки")

        try:
            drone.land()
            airborne = False
        except Exception as landing_error:
            print(
                "Ошибка аварийной посадки:",
                landing_error,
            )

finally:
    print("Освобождение ресурсов")

    stop_vision()

    if model is not None:
        try:
            model.release()
        except Exception as error:
            print(
                "Ошибка освобождения RKNN:",
                error,
            )

    if viewer is not None:
        try:
            viewer.close()
        except Exception as error:
            print(
                "Ошибка закрытия трансляции:",
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

    if drone is not None:
        try:
            drone.close_connection()
        except Exception as error:
            print(
                "Ошибка закрытия Pioneer:",
                error,
            )

    print("Ресурсы освобождены")