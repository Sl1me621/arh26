from __future__ import annotations

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

FIELD_WIDTH_M = 7.975
FIELD_HEIGHT_M = 5.875

MAP_IMAGE_PATH = Path(
    r"C:\Users\1234\arh26\3_track\image.jpg"
)

WINDOW_NAME = (
    "Geoscan Archipelag 2026 - Control Station"
)

# Размер отображаемого окна.
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720


# ============================================================
# ЦВЕТА СТАНЦИЙ В ФОРМАТЕ BGR
# ============================================================

COLOR_MAP = {
    "обнаружена": (0, 255, 255),
    "panel": (0, 255, 0),
    "dust": (0, 165, 255),
    "исправна": (0, 255, 0),
    "покрыта пылью": (0, 165, 255),
    "неисправна": (0, 0, 255),
}

UNKNOWN_COLOR = (255, 0, 255)


# ============================================================
# FLASK И ОБЩЕЕ СОСТОЯНИЕ
# ============================================================

app = Flask(__name__)

data_lock = threading.Lock()
stop_event = threading.Event()

# Формат:
#
# {
#     1: {
#         "id": 1,
#         "x": 0.53,
#         "y": 2.15,
#         "status": "panel",
#         "confidence": 0.91
#     }
# }
detected_stations: dict[int, dict] = {}

map_needs_update = threading.Event()
map_needs_update.set()


# ============================================================
# ГОМОГРАФИЯ
# ============================================================



def draw_meter_grid(
    frame: np.ndarray,
    step_m: float = 1.0,
) -> None:
    """
    Рисует сетку карты с шагом 1 метр.

    Координата X возрастает справа налево,
    координата Y — сверху вниз, как в текущем
    преобразовании map_meters_to_pixels().
    """

    image_height, image_width = frame.shape[:2]

    grid_color = (180, 180, 180)
    axis_color = (255, 255, 255)
    text_color = (255, 255, 255)

    overlay = frame.copy()

    # Вертикальные линии X: 0...8 метров.
    x_m = 0.0

    while x_m <= FIELD_WIDTH_M + 1e-6:
        pixel_x, _ = map_meters_to_pixels(
            x_m=x_m,
            y_m=0.0,
            image_width=image_width,
            image_height=image_height,
        )

        pixel_x = max(
            0,
            min(pixel_x, image_width - 1),
        )

        color = (
            axis_color
            if abs(x_m) < 1e-6
            else grid_color
        )

        thickness = (
            2
            if abs(x_m) < 1e-6
            else 1
        )

        cv2.line(
            overlay,
            (pixel_x, 0),
            (pixel_x, image_height - 1),
            color,
            thickness,
            cv2.LINE_AA,
        )

        label = f"{x_m:g}"

        cv2.putText(
            overlay,
            label,
            (
                max(2, pixel_x - 7),
                20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

        x_m += step_m

    # Горизонтальные линии Y: 0...6 метров.
    y_m = 0.0

    while y_m <= FIELD_HEIGHT_M + 1e-6:
        _, pixel_y = map_meters_to_pixels(
            x_m=0.0,
            y_m=y_m,
            image_width=image_width,
            image_height=image_height,
        )

        pixel_y = max(
            0,
            min(pixel_y, image_height - 1),
        )

        color = (
            axis_color
            if abs(y_m) < 1e-6
            else grid_color
        )

        thickness = (
            2
            if abs(y_m) < 1e-6
            else 1
        )

        cv2.line(
            overlay,
            (0, pixel_y),
            (image_width - 1, pixel_y),
            color,
            thickness,
            cv2.LINE_AA,
        )

        label = f"{y_m:g}"

        cv2.putText(
            overlay,
            label,
            (5, max(15, pixel_y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            text_color,
            1,
            cv2.LINE_AA,
        )

        y_m += step_m

    # Полупрозрачная сетка поверх карты.
    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame,
    )
def map_meters_to_pixels(
    x_m: float,
    y_m: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int]:
    """
    Переводит координаты карты 8×6 м в пиксели.

    (0, 0) — верхний правый угол.
    X растёт справа налево.
    Y растёт сверху вниз.
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


# ============================================================
# СИСТЕМЫ КООРДИНАТ
# ============================================================

FIELD_WIDTH_M = 8.0
FIELD_HEIGHT_M = 6.0

# Локальная точка старта дрона (0, 0)
# на карте находится в координате (4.5, 1.5).
DRONE_START_MAP_X = 1.5
DRONE_START_MAP_Y = 4.5


def drone_to_map_coordinates(
    x_drone: float,
    y_drone: float,
) -> tuple[float, float]:
    """
    Переводит локальные координаты дрона
    в метрические координаты карты.

    Карта:
        X растёт справа налево.
        Y растёт сверху вниз.

    Дрон:
        локальная точка (0, 0)
        соответствует карте (4.5, 1.5).
    """

    x_map = DRONE_START_MAP_X - y_drone
    y_map = DRONE_START_MAP_Y + x_drone

    return x_map, y_map


# ============================================================
# ПРОВЕРКА ВХОДНЫХ ДАННЫХ
# ============================================================

def parse_station_data(data) -> dict:
    if not isinstance(data, dict):
        raise ValueError(
            "Тело запроса должно быть JSON-объектом"
        )

    required_fields = {
        "id",
        "x",
        "y",
    }

    missing_fields = required_fields - data.keys()

    if missing_fields:
        raise ValueError(
            "Отсутствуют поля: "
            + ", ".join(sorted(missing_fields))
        )

    station_id = int(data["id"])
    x_coordinate = float(data["x"])
    y_coordinate = float(data["y"])

    if not np.isfinite(x_coordinate):
        raise ValueError("Координата x некорректна")

    if not np.isfinite(y_coordinate):
        raise ValueError("Координата y некорректна")

    status = str(
        data.get("status", "обнаружена")
    )

    confidence = data.get("confidence")

    if confidence is not None:
        confidence = float(confidence)

    confirmations = int(
        data.get("confirmations", 1)
    )

    return {
        "id": station_id,
        "x": x_coordinate,
        "y": y_coordinate,
        "status": status,
        "confidence": confidence,
        "confirmations": confirmations,
        "received_at": time.time(),
    }


# ============================================================
# HTTP-МАРШРУТЫ
# ============================================================

@app.route("/drone", methods=["POST"])
def receive_drone_data():
    """
    Принимает JSON:

    {
        "id": 1,
        "x": 0.53,
        "y": 2.15,
        "status": "panel",
        "confidence": 0.91,
        "confirmations": 5
    }
    """

    try:
        request_data = request.get_json(
            force=True
        )

        station = parse_station_data(
            request_data
        )

    except Exception as error:
        print(
            "[ОШИБКА ПРИЕМА]",
            type(error).__name__,
            error,
        )

        return jsonify({
            "status": "error",
            "message": str(error),
        }), 400

    station_id = station["id"]

    with data_lock:
        is_new_station = (
            station_id not in detected_stations
        )

        detected_stations[station_id] = station

    map_needs_update.set()

    action = (
        "добавлена"
        if is_new_station
        else "обновлена"
    )

    print(
        f"[HTTP] Станция №{station_id} {action}: "
        f"x={station['x']:.3f}, "
        f"y={station['y']:.3f}, "
        f"status={station['status']}, "
        f"confidence={station['confidence']}"
    )

    return jsonify({
        "status": "ok",
        "message": "station received",
        "station_id": station_id,
        "stations_count": len(detected_stations),
    }), 200


@app.route("/stations", methods=["GET"])
def get_stations():
    with data_lock:
        stations = list(
            detected_stations.values()
        )

    return jsonify({
        "count": len(stations),
        "stations": stations,
    }), 200


@app.route("/clear", methods=["POST"])
def clear_stations():
    with data_lock:
        detected_stations.clear()

    map_needs_update.set()

    print("[HTTP] Список станций очищен")

    return jsonify({
        "status": "ok",
        "message": "stations cleared",
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
    }), 200


def run_http_server():
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
    font_scale: float = 0.55,
) -> None:
    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_width, text_height), baseline = (
        cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
    )

    cv2.rectangle(
        frame,
        (x - 4, y - text_height - 6),
        (
            x + text_width + 4,
            y + baseline + 4,
        ),
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


def draw_station(
    frame: np.ndarray,
    station: dict,
) -> None:
    image_height, image_width = frame.shape[:2]

    station_id = int(station["id"])
    x_drone = float(station["x"])
    y_drone = float(station["y"])

    status = str(
        station.get("status", "обнаружена")
    )

    confidence = station.get("confidence")
    confirmations = station.get(
        "confirmations",
        1,
    )

    point_color = COLOR_MAP.get(
        status,
        UNKNOWN_COLOR,
    )

    x_map, y_map = drone_to_map_coordinates(
        x_drone,
        y_drone,
    )

    pixel_x, pixel_y = map_meters_to_pixels(
        x_map,
        y_map,
        image_width,
        image_height,
    )

    # Проверяем, попала ли точка внутрь изображения.
    if not (
        0 <= pixel_x < image_width
        and 0 <= pixel_y < image_height
    ):
        print(
            f"[ПРЕДУПРЕЖДЕНИЕ] "
            f"Станция №{station_id} вне карты: "
            f"pixel=({pixel_x}, {pixel_y})"
        )
        return

    # Чёрная внешняя окружность.
    cv2.circle(
        frame,
        (pixel_x, pixel_y),
        14,
        (0, 0, 0),
        -1,
    )

    # Цвет станции.
    cv2.circle(
        frame,
        (pixel_x, pixel_y),
        11,
        point_color,
        -1,
    )

    # Центральная белая точка.
    cv2.circle(
        frame,
        (pixel_x, pixel_y),
        3,
        (255, 255, 255),
        -1,
    )

    first_line = (
        f"#{station_id} {status}"
    )

    second_line = (
        f"x={x_drone:.2f}, y={y_drone:.2f}"
    )

    if confidence is not None:
        second_line += (
            f", conf={float(confidence):.2f}"
        )

    third_line = (
        f"confirmations={confirmations}"
    )

    label_x = pixel_x + 18
    label_y = pixel_y - 16

    # Если справа места мало — подпись слева.
    if label_x + 300 >= image_width:
        label_x = max(
            5,
            pixel_x - 300,
        )

    # Если сверху места мало — подпись снизу.
    if label_y - 45 < 0:
        label_y = pixel_y + 28

    draw_text_with_background(
        frame,
        first_line,
        (label_x, label_y),
        font_scale=0.6,
    )

    draw_text_with_background(
        frame,
        second_line,
        (label_x, label_y + 22),
        font_scale=0.5,
    )

    draw_text_with_background(
        frame,
        third_line,
        (label_x, label_y + 44),
        font_scale=0.5,
    )

def draw_drone_start(
    frame: np.ndarray,
) -> None:
    """
    Рисует на карте точку, соответствующую
    локальной координате дрона (0, 0).
    """

    image_height, image_width = frame.shape[:2]

    pixel_x, pixel_y = map_meters_to_pixels(
        DRONE_START_MAP_X,
        DRONE_START_MAP_Y,
        image_width,
        image_height,
    )

    # Внешняя окружность.
    cv2.circle(
        frame,
        (pixel_x, pixel_y),
        16,
        (0, 0, 0),
        -1,
    )

    # Синяя точка старта.
    cv2.circle(
        frame,
        (pixel_x, pixel_y),
        12,
        (255, 0, 0),
        -1,
    )

    # Белый центр.
    cv2.circle(
        frame,
        (pixel_x, pixel_y),
        3,
        (255, 255, 255),
        -1,
    )

    label = (
        "START: drone (0,0), map (4.5,1.5)"
    )

    cv2.putText(
        frame,
        label,
        (pixel_x + 18, pixel_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        label,
        (pixel_x + 18, pixel_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

def draw_header(
    frame: np.ndarray,
    stations_count: int,
) -> None:
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (390, 95),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.65,
        frame,
        0.35,
        0,
        frame,
    )

    cv2.putText(
        frame,
        "Geoscan Archipelag 2026",
        (22, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        f"Detected stations: {stations_count}",
        (22, 74),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def build_map_frame(
    original_map: np.ndarray,
) -> np.ndarray:
    output_frame = original_map.copy()

    draw_meter_grid(
        output_frame,
        step_m=1.0,
    )

    draw_drone_start(
        output_frame,
    )

    with data_lock:
        stations_snapshot = [
            dict(station)
            for station in detected_stations.values()
        ]

    for station in stations_snapshot:
        draw_station(
            output_frame,
            station,
        )

    draw_header(
        output_frame,
        len(stations_snapshot),
    )

    return output_frame

def display_loop():
    original_map = cv2.imread(
        str(MAP_IMAGE_PATH)
    )

    if original_map is None:
        raise FileNotFoundError(
            f"Не удалось открыть карту: "
            f"{MAP_IMAGE_PATH}"
        )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    working_map = build_map_frame(
        original_map
    )

    print("Окно карты запущено")
    print("ESC — завершить программу")
    print("C — очистить станции")
    print("S — сохранить текущую карту")

    while not stop_event.is_set():
        if map_needs_update.is_set():
            working_map = build_map_frame(
                original_map
            )

            map_needs_update.clear()

        cv2.imshow(
            WINDOW_NAME,
            working_map,
        )

        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            stop_event.set()
            break

        if key in (ord("c"), ord("C")):
            with data_lock:
                detected_stations.clear()

            map_needs_update.set()

            print("[КАРТА] Станции очищены")

        if key in (ord("s"), ord("S")):
            output_path = Path(
                "detected_stations_map.jpg"
            )

            cv2.imwrite(
                str(output_path),
                working_map,
            )

            print(
                f"[КАРТА] Изображение сохранено: "
                f"{output_path.resolve()}"
            )

    cv2.destroyAllWindows()


# ============================================================
# ЗАПУСК
# ============================================================

def main():
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