from pioneer_sdk import Pioneer, Camera

import cv2
import numpy as np
import threading
import time


# ============================================================
# ПАРАМЕТРЫ ПОЛЁТА
# ============================================================

ALTITUDE = 1.8
HOVER_TIME = 5.0
POINT_TIMEOUT = 25.0

waypoints = [
    (13.4, 1.2),
    (13.4, 5.0),
]


# ============================================================
# ПАРАМЕТРЫ ARUCO
# ============================================================

# Должен совпадать со словарём, которым создана метка.
ARUCO_DICTIONARY_TYPE = cv2.aruco.DICT_4X4_50

# None — искать любую метку.
# Например, 15 — реагировать только на ArUco ID 15.
TARGET_ARUCO_ID = None

# Сколько кадров подряд должна определиться одна метка.
ARUCO_CONFIRM_FRAMES = 3


# ============================================================
# ПАРАМЕТРЫ ЦЕНТРИРОВАНИЯ
# ============================================================

CENTER_KP = 0.35

# Максимальная горизонтальная скорость при центрировании.
MAX_CENTER_SPEED = 0.25

# Допустимое отклонение от центра — доля размера кадра.
CENTER_TOLERANCE_RATIO = 0.03

# Сколько времени маркер должен находиться в центре.
CENTER_STABLE_TIME = 1.0

# Максимальное время центрирования.
CENTER_TIMEOUT = 20.0

# Через сколько секунд информация о маркере устаревает.
MARKER_STALE_TIME = 0.35

# Частота отправки команд скорости.
CONTROL_PERIOD = 0.05

# Время зависания после успешного центрирования.
MARKER_HOVER_TIME = 5.0


# ============================================================
# НАПРАВЛЕНИЯ ОСЕЙ КАМЕРЫ
# ============================================================

# При камере, направленной вниз:
#
# маркер справа в кадре -> дрон должен двигаться вправо;
# маркер снизу в кадре -> дрон должен двигаться назад.
#
# Если дрон движется ОТ метки, поменяй знак соответствующего
# коэффициента с 1.0 на -1.0 или наоборот.

IMAGE_X_TO_BODY_Y_SIGN = 1.0
IMAGE_Y_TO_BODY_X_SIGN = -1.0


# ============================================================
# СОЗДАНИЕ ОБЪЕКТОВ
# ============================================================

drone = Pioneer(
    ip="127.0.0.1",
    mavlink_port=8001,
    log_connection=True,
)

camera = Camera(
    ip="127.0.0.1",
    port=18001,
    timeout=1.0,
    log_connection=True,
)


# ============================================================
# СОБЫТИЯ И ОБЩИЕ ДАННЫЕ
# ============================================================

camera_stop_event = threading.Event()
camera_ready_event = threading.Event()

# Разрешает камере прервать маршрут при обнаружении метки.
marker_search_enabled = threading.Event()

# Устанавливается после подтверждённого обнаружения.
marker_found_event = threading.Event()

# Аварийная остановка программы, например клавишей Q.
abort_event = threading.Event()

marker_lock = threading.Lock()

marker_observation = {
    "id": None,
    "center": None,
    "frame_size": None,
    "area": 0.0,
    "last_seen": 0.0,
}


# ============================================================
# СОЗДАНИЕ ДЕТЕКТОРА ARUCO
# ============================================================

if not hasattr(cv2, "aruco"):
    raise RuntimeError(
        "В установленном OpenCV нет модуля aruco. "
        "Установи opencv-contrib-python."
    )

aruco_dictionary = cv2.aruco.getPredefinedDictionary(
    ARUCO_DICTIONARY_TYPE
)

if hasattr(cv2.aruco, "DetectorParameters"):
    aruco_parameters = cv2.aruco.DetectorParameters()
else:
    aruco_parameters = cv2.aruco.DetectorParameters_create()

if hasattr(cv2.aruco, "ArucoDetector"):
    aruco_detector = cv2.aruco.ArucoDetector(
        aruco_dictionary,
        aruco_parameters,
    )
else:
    aruco_detector = None


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def detect_aruco(frame):
    """
    Находит ArUco-метки.

    Возвращает информацию о выбранной метке:
    ID, центр, углы и площадь.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if aruco_detector is not None:
        corners, ids, rejected = aruco_detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray,
            aruco_dictionary,
            parameters=aruco_parameters,
        )

    if ids is None or len(ids) == 0:
        return None

    ids = ids.flatten()

    candidates = []

    for index, marker_id in enumerate(ids):
        marker_id = int(marker_id)

        if (
            TARGET_ARUCO_ID is not None
            and marker_id != TARGET_ARUCO_ID
        ):
            continue

        points = corners[index].reshape(4, 2)

        center_x = float(np.mean(points[:, 0]))
        center_y = float(np.mean(points[:, 1]))

        area = abs(cv2.contourArea(points.astype(np.float32)))

        candidates.append({
            "id": marker_id,
            "center": (center_x, center_y),
            "corners": points,
            "area": area,
        })

    if not candidates:
        return None

    # Если в кадре несколько меток, выбираем самую крупную.
    return max(
        candidates,
        key=lambda candidate: candidate["area"],
    )


def draw_marker_information(frame, detection):
    """Рисует рамку, ID и центр ArUco."""

    height, width = frame.shape[:2]

    frame_center = (
        int(width / 2),
        int(height / 2),
    )

    # Центр кадра.
    cv2.drawMarker(
        frame,
        frame_center,
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=25,
        thickness=2,
    )

    if detection is None:
        cv2.putText(
            frame,
            "ArUco: not found",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        return

    points = detection["corners"].astype(np.int32)

    marker_center = (
        int(detection["center"][0]),
        int(detection["center"][1]),
    )

    cv2.polylines(
        frame,
        [points],
        True,
        (0, 255, 0),
        2,
    )

    cv2.circle(
        frame,
        marker_center,
        6,
        (0, 0, 255),
        -1,
    )

    # Линия между центром кадра и центром метки.
    cv2.line(
        frame,
        frame_center,
        marker_center,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        (
            f"ArUco ID: {detection['id']} "
            f"center: {marker_center}"
        ),
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )


def get_marker_observation():
    """Возвращает копию последних данных о маркере."""

    with marker_lock:
        return marker_observation.copy()


# ============================================================
# ПОТОК КАМЕРЫ
# ============================================================

def camera_stream():
    print("Подключение к камере симулятора...")

    last_marker_id = None
    confirmation_count = 0
    detection_announced = False

    try:
        camera.connect()
        camera_ready_event.set()

        print("Видеопоток запущен")

        while not camera_stop_event.is_set():
            try:
                frame = camera.get_cv_frame()

                if frame is None:
                    time.sleep(0.01)
                    continue

                detection = detect_aruco(frame)

                if detection is not None:
                    marker_id = detection["id"]

                    if marker_id == last_marker_id:
                        confirmation_count += 1
                    else:
                        last_marker_id = marker_id
                        confirmation_count = 1
                        detection_announced = False

                    height, width = frame.shape[:2]

                    with marker_lock:
                        marker_observation.update({
                            "id": marker_id,
                            "center": detection["center"],
                            "frame_size": (width, height),
                            "area": detection["area"],
                            "last_seen": time.monotonic(),
                        })

                    if (
                        marker_search_enabled.is_set()
                        and confirmation_count
                        >= ARUCO_CONFIRM_FRAMES
                    ):
                        marker_found_event.set()

                        if not detection_announced:
                            print(
                                f"Найдена ArUco ID {marker_id}, "
                                f"центр: {detection['center']}"
                            )
                            detection_announced = True

                else:
                    last_marker_id = None
                    confirmation_count = 0
                    detection_announced = False

                display_frame = frame.copy()

                draw_marker_information(
                    display_frame,
                    detection,
                )

                if marker_found_event.is_set():
                    status = "MARKER FOUND"
                    status_color = (0, 255, 255)
                elif marker_search_enabled.is_set():
                    status = (
                        f"SEARCHING: "
                        f"{confirmation_count}/"
                        f"{ARUCO_CONFIRM_FRAMES}"
                    )
                    status_color = (255, 255, 0)
                else:
                    status = "SEARCH DISABLED"
                    status_color = (150, 150, 150)

                cv2.putText(
                    display_frame,
                    status,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )

                cv2.imshow(
                    "Pioneer simulator camera",
                    display_frame,
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    print("Нажата Q — аварийная остановка")
                    abort_event.set()
                    camera_stop_event.set()
                    break

            except Exception as error:
                print("Ошибка видеопотока:", error)
                time.sleep(0.2)

    except Exception as error:
        print("Не удалось подключиться к камере:", error)
        abort_event.set()

    finally:
        camera_ready_event.set()
        cv2.destroyAllWindows()
        print("Видеопоток остановлен")


# ============================================================
# УПРАВЛЕНИЕ ДРОНОМ
# ============================================================

def wait_for_connection(timeout=15.0):
    deadline = time.monotonic() + timeout

    while not drone.connected():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "Нет подключения к симулятору"
            )

        if abort_event.is_set():
            raise KeyboardInterrupt

        time.sleep(0.2)

    print("Соединение с симулятором установлено")


def send_zero_speed():
    """Отправляет одну команду нулевой скорости."""

    drone.set_manual_speed_body_fixed(
        vx=0.0,
        vy=0.0,
        vz=0.0,
        yaw_rate=0.0,
    )


def stop_drone_motion(duration=0.6):
    """
    Прерывает движение к заданной точке.

    Несколько раз отправляет нулевую скорость,
    чтобы переключить управление с полёта к точке
    на ручное управление скоростью.
    """

    print("Остановка текущего полётного задания")

    end_time = time.monotonic() + duration

    while time.monotonic() < end_time:
        send_zero_speed()
        time.sleep(CONTROL_PERIOD)


def wait_for_point(
    timeout=POINT_TIMEOUT,
    stop_on_marker=True,
):
    """
    Ждёт достижения точки.

    Возвращает:
    True  — точка достигнута;
    False — во время полёта найдена ArUco.
    """

    deadline = time.monotonic() + timeout

    while True:
        if abort_event.is_set():
            raise KeyboardInterrupt

        if (
            stop_on_marker
            and marker_found_event.is_set()
        ):
            stop_drone_motion()
            return False

        if drone.point_reached():
            return True

        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Дрон не достиг точки за {timeout} секунд"
            )

        time.sleep(0.05)


def go_to_point(
    x,
    y,
    z=ALTITUDE,
    yaw=0.0,
    stop_on_marker=True,
):
    print(
        f"Полёт в точку: "
        f"x={x}, y={y}, z={z}"
    )

    command_result = drone.go_to_local_point(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
    )

    if not command_result:
        raise RuntimeError(
            "Симулятор не принял точку"
        )

    return wait_for_point(
        timeout=POINT_TIMEOUT,
        stop_on_marker=stop_on_marker,
    )


def hover(seconds, stop_on_marker=True):
    """
    Зависание.

    Возвращает False, если во время зависания
    обнаружена ArUco.
    """

    print(f"Зависание: {seconds} секунд")

    end_time = time.monotonic() + seconds

    while time.monotonic() < end_time:
        if abort_event.is_set():
            raise KeyboardInterrupt

        if (
            stop_on_marker
            and marker_found_event.is_set()
        ):
            stop_drone_motion()
            return False

        time.sleep(0.05)

    return True


def fly_through_points(points):
    """
    Выполняет маршрут.

    Возвращает False, если маршрут был прерван
    обнаружением ArUco.
    """

    for number, (x, y) in enumerate(
        points,
        start=1,
    ):
        print(
            f"Точка {number}/{len(points)}"
        )

        point_reached = go_to_point(
            x=x,
            y=y,
            z=ALTITUDE,
            yaw=0.0,
            stop_on_marker=True,
        )

        if not point_reached:
            return False

        hover_completed = hover(
            HOVER_TIME,
            stop_on_marker=True,
        )

        if not hover_completed:
            return False

    return True


# ============================================================
# ЦЕНТРИРОВАНИЕ НАД ARUCO
# ============================================================

def center_over_marker():
    """
    Центрирует дрон над найденной ArUco.

    Используется простой пропорциональный регулятор:
    чем дальше метка от центра кадра,
    тем выше скорость движения.
    """

    print("Начато центрирование над ArUco")

    stop_drone_motion()

    deadline = time.monotonic() + CENTER_TIMEOUT
    centered_since = None

    while time.monotonic() < deadline:
        if abort_event.is_set():
            raise KeyboardInterrupt

        observation = get_marker_observation()

        marker_center = observation["center"]
        frame_size = observation["frame_size"]
        last_seen = observation["last_seen"]

        marker_is_fresh = (
            marker_center is not None
            and frame_size is not None
            and time.monotonic() - last_seen
            <= MARKER_STALE_TIME
        )

        if not marker_is_fresh:
            centered_since = None
            send_zero_speed()
            time.sleep(CONTROL_PERIOD)
            continue

        marker_x, marker_y = marker_center
        frame_width, frame_height = frame_size

        frame_center_x = frame_width / 2.0
        frame_center_y = frame_height / 2.0

        error_x_px = marker_x - frame_center_x
        error_y_px = marker_y - frame_center_y

        tolerance_x = max(
            10.0,
            frame_width * CENTER_TOLERANCE_RATIO,
        )

        tolerance_y = max(
            10.0,
            frame_height * CENTER_TOLERANCE_RATIO,
        )

        inside_center = (
            abs(error_x_px) <= tolerance_x
            and abs(error_y_px) <= tolerance_y
        )

        if inside_center:
            send_zero_speed()

            if centered_since is None:
                centered_since = time.monotonic()
                print("Метка попала в область центра")

            centered_time = (
                time.monotonic() - centered_since
            )

            if centered_time >= CENTER_STABLE_TIME:
                print(
                    "Центрирование завершено. "
                    "Зависание над меткой 5 секунд"
                )

                hold_end = (
                    time.monotonic()
                    + MARKER_HOVER_TIME
                )

                while time.monotonic() < hold_end:
                    if abort_event.is_set():
                        raise KeyboardInterrupt

                    send_zero_speed()
                    time.sleep(CONTROL_PERIOD)

                return True

        else:
            centered_since = None

            # Нормализация ошибки примерно в диапазон [-1; 1].
            error_x_normalized = (
                error_x_px / frame_center_x
            )

            error_y_normalized = (
                error_y_px / frame_center_y
            )

            # Вертикальная координата изображения
            # управляет движением вперёд/назад.
            vx = (
                IMAGE_Y_TO_BODY_X_SIGN
                * CENTER_KP
                * error_y_normalized
            )

            # Горизонтальная координата изображения
            # управляет движением вправо/влево.
            vy = (
                IMAGE_X_TO_BODY_Y_SIGN
                * CENTER_KP
                * error_x_normalized
            )

            vx = clamp(
                vx,
                -MAX_CENTER_SPEED,
                MAX_CENTER_SPEED,
            )

            vy = clamp(
                vy,
                -MAX_CENTER_SPEED,
                MAX_CENTER_SPEED,
            )

            command_result = (
                drone.set_manual_speed_body_fixed(
                    vx=vx,
                    vy=vy,
                    vz=0.0,
                    yaw_rate=0.0,
                )
            )

            if not command_result:
                print(
                    "Предупреждение: команда скорости "
                    "не была принята"
                )

        time.sleep(CONTROL_PERIOD)

    stop_drone_motion()

    print(
        f"Не удалось отцентрироваться "
        f"за {CENTER_TIMEOUT} секунд"
    )

    return False


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================

camera_thread = threading.Thread(
    target=camera_stream,
    daemon=True,
)

camera_thread.start()

flight_started = False

try:
    wait_for_connection()

    # Ждём подключения камеры.
    if not camera_ready_event.wait(timeout=10.0):
        raise TimeoutError(
            "Камера не подключилась за 10 секунд"
        )

    if abort_event.is_set():
        raise RuntimeError(
            "Не удалось запустить видеопоток"
        )

    if not drone.arm():
        raise RuntimeError(
            "Не удалось запустить моторы"
        )

    flight_started = True

    time.sleep(1.0)

    if not drone.takeoff():
        raise RuntimeError(
            "Не удалось выполнить взлёт"
        )

    time.sleep(3.0)

    # Начинаем реагировать на найденные метки.
    marker_search_enabled.set()

    print("Поиск ArUco включён")

    mission_completed = go_to_point(
        x=14.0,
        y=1.0,
        z=ALTITUDE,
        yaw=0.0,
        stop_on_marker=True,
    )

    if mission_completed:
        mission_completed = fly_through_points(
            waypoints
        )

    if not mission_completed:
        print(
            "Маршрут прерван из-за обнаружения ArUco"
        )

        centered = center_over_marker()

        if centered:
            print(
                "Дрон успешно выровнен "
                "относительно маркера"
            )
        else:
            print(
                "Центрирование завершилось неудачно"
            )

    else:
        print(
            "Маршрут завершён, "
            "ArUco не обнаружена"
        )

    # После обнаружения метки маршрут уже не продолжается.
    marker_search_enabled.clear()
    marker_found_event.clear()

    print("Возвращение в стартовую точку")

    go_to_point(
        x=0.0,
        y=0.0,
        z=ALTITUDE,
        yaw=0.0,
        stop_on_marker=False,
    )

    print("Посадка")

    drone.land()
    time.sleep(5.0)

    flight_started = False

except KeyboardInterrupt:
    print("Остановка программы")

    if flight_started:
        drone.land()
        time.sleep(3.0)

except Exception as error:
    print(
        f"Ошибка: {type(error).__name__}: {error}"
    )

    if flight_started:
        drone.land()
        time.sleep(3.0)

finally:
    marker_search_enabled.clear()
    camera_stop_event.set()

    if camera_thread.is_alive():
        camera_thread.join(timeout=2.0)

    try:
        camera.disconnect()
    except Exception:
        pass

    cv2.destroyAllWindows()
    drone.close_connection()

    print("Программа завершена")