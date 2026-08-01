import os
import cv2
import json
import threading
import numpy as np
from flask import Flask, request

# --- Настройки ---
FIELD_WIDTH_M  = 7.975   
FIELD_HEIGHT_M = 5.875   
MAP_IMAGE_PATH = "C:/Users/1234/a26/map.jpg"
PORT = 5001

COLOR_MAP = {
    'исправна': (0, 255, 0),    # Зеленый
    'неисправна': (0, 0, 255)   # Красный
}

app = Flask(__name__)
lock = threading.Lock()

# Глобальные переменные для работы с картой
refresh_map_flag = True
detected_stations = []
working_map = None

# --- Математика гомографии ---
def drone_to_right_down(x_d, y_d):
    return (y_d, x_d)

def compute_homography_drone_to_map(drone_corners):
    tl_rd = drone_to_right_down(*drone_corners['tl'])
    bl_rd = drone_to_right_down(*drone_corners['bl'])
    br_rd = drone_to_right_down(*drone_corners['br'])
    tr_rd = drone_to_right_down(*drone_corners['tr'])

    src = np.array([tr_rd, tl_rd, bl_rd, br_rd], dtype=np.float32)
    dst = np.array([
        [0.0,           0.0],
        [FIELD_WIDTH_M, 0.0],
        [FIELD_WIDTH_M, FIELD_HEIGHT_M],
        [0.0,           FIELD_HEIGHT_M],
    ], dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)

def transform_point(H, x_d, y_d):
    pt_rd = drone_to_right_down(x_d, y_d)
    pt_np = np.array([[pt_rd]], dtype=np.float32)
    warped = cv2.perspectiveTransform(pt_np, H)
    return warped[0][0]

def map_meters_to_pixels(x_m, y_m, img_w, img_h):
    px = int(round(img_w - (x_m / FIELD_WIDTH_M) * img_w))
    py = int(round((y_m / FIELD_HEIGHT_M) * img_h))
    return px, py

# Вычисляем матрицу один раз при запуске
drone_corners = {
    'tl': (-0.8, -5.0), 'bl': ( 4.5, -5.0),
    'br': ( 4.5,  2.0), 'tr': (-0.8,  2.2)
}
H = compute_homography_drone_to_map(drone_corners)


# --- FLASK СЕРВЕР ---
@app.route("/drone", methods=["POST"])
def receive_drone_data():
    """Принимает JSON от дрона вида: {"x": 1.2, "y": 3.4, "status": "исправна"}"""
    global refresh_map_flag
    
    # Получаем данные из POST запроса
    try:
        data = request.get_json(force=True)
    except Exception:
        data = request.get_data(as_text=True)
        print(f"[ОШИБКА ПАРСИНГА] Получен не JSON: {data}")
        return "Invalid JSON", 400

    # Блокируем данные для безопасного добавления
    with lock:
        if 'x' in data and 'y' in data:
            detected_stations.append(data)
            print(f"[HTTP ПРИЕМ] Станция: {data}")
            refresh_map_flag = True

    return "Data received", 200


# --- ПОТОК ОТРИСОВКИ OPENCV ---
def display_loop():
    global refresh_map_flag, working_map
    
    original_map = cv2.imread(MAP_IMAGE_PATH)
    if original_map is None:
        print(f"[ОШИБКА] Не найдена карта {MAP_IMAGE_PATH}")
        os._exit(1)
        
    h_img, w_img = original_map.shape[:2]
    
    window_name = "Geoscan Archipelag 2026 - HTTP Control Station"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    while True:
        if refresh_map_flag:
            with lock:
                working_map = original_map.copy()
                # Рисуем все станции
                for station in detected_stations:
                    x_d, y_d = station['x'], station['y']
                    status = station.get('status', 'исправна')
                    bgr = COLOR_MAP.get(status, (255, 0, 255))
                    
                    x_m, y_m = transform_point(H, x_d, y_d)
                    px, py = map_meters_to_pixels(x_m, y_m, w_img, h_img)
                    
                    cv2.circle(working_map, (px, py), 12, (0,0,0), -1)
                    cv2.circle(working_map, (px, py), 10, bgr, -1)
                    label = f"{status} ({x_d:.1f}, {y_d:.1f})"
                    cv2.putText(working_map, label, (px + 15, py + 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
                
            cv2.imshow(window_name, working_map)
            refresh_map_flag = False

        if cv2.waitKey(50) & 0xFF == 27: # Выход по ESC
            break

    cv2.destroyAllWindows()
    os._exit(0)


if __name__ == '__main__':
    # Запуск OpenCV в фоновом потоке
    display = threading.Thread(target=display_loop, daemon=True)
    display.start()
    
    # Запуск Flask сервера
    print(f"[СТАРТ] HTTP сервер запущен на порту {PORT}...")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)