from pioneer_sdk2 import Camera, ServoCamera

from flask import Flask, Response, jsonify
import cv2
import threading
import time


SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001

CAMERA_ANGLE = -80
JPEG_QUALITY = 80


app = Flask(__name__)

camera = Camera()
servo_camera = ServoCamera()

# Установлено ли разрешение на трансляцию.
stream_enabled = threading.Event()

# Сигнал полного завершения программы.
program_stopped = threading.Event()

# Последний закодированный JPEG-кадр.
latest_jpeg = None
frame_number = 0

# Используется для безопасной передачи кадров между потоками.
frame_condition = threading.Condition()


def camera_capture_loop():
    """
    Получает кадры с камеры и кодирует их в JPEG.

    Поток работает постоянно, но получает кадры только после
    команды POST /camera/start.
    """

    global latest_jpeg
    global frame_number

    print("Поток обработки камеры создан")

    while not program_stopped.is_set():

        # Ждём команду запуска трансляции.
        if not stream_enabled.wait(timeout=0.2):
            continue

        try:
            frame = camera.get_cv_frame(timeout=5.0)

            if frame is None:
                print("Камера не вернула кадр")
                continue

            encode_success, encoded_frame = cv2.imencode(
                ".jpg",
                frame,
                [
                    int(cv2.IMWRITE_JPEG_QUALITY),
                    JPEG_QUALITY
                ]
            )

            if not encode_success:
                print("Не удалось закодировать кадр")
                continue

            with frame_condition:
                latest_jpeg = encoded_frame.tobytes()
                frame_number += 1

                # Сообщаем подключённым клиентам,
                # что появился новый кадр.
                frame_condition.notify_all()

        except Exception as error:
            print("Ошибка получения кадра:", error)
            time.sleep(0.5)

    print("Поток обработки камеры завершён")


def generate_mjpeg_stream():
    """
    Генерирует MJPEG-поток для HTTP-клиента.
    """

    last_frame_number = -1

    while stream_enabled.is_set() and not program_stopped.is_set():

        with frame_condition:

            # Ждём новый кадр.
            while (
                frame_number == last_frame_number
                and stream_enabled.is_set()
                and not program_stopped.is_set()
            ):
                frame_condition.wait(timeout=2.0)

            if not stream_enabled.is_set():
                break

            if program_stopped.is_set():
                break

            if latest_jpeg is None:
                continue

            jpeg = latest_jpeg
            last_frame_number = frame_number

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: "
            + str(len(jpeg)).encode()
            + b"\r\n\r\n"
            + jpeg
            + b"\r\n"
        )


@app.route("/camera/start", methods=["POST"])
def start_camera_stream():
    """
    Запускает получение и передачу кадров.
    """

    if stream_enabled.is_set():
        return jsonify({
            "status": "already_running",
            "message": "Camera stream is already running"
        }), 200

    try:
        angle_result = servo_camera.set_angle(CAMERA_ANGLE)

        print(
            f"Камера установлена на угол {CAMERA_ANGLE}. "
            f"Результат: {angle_result}"
        )

        stream_enabled.set()

        print("Получена команда запуска трансляции")

        return jsonify({
            "status": "started",
            "message": "Camera stream started",
            "stream_url": "/camera/stream",
            "camera_angle": CAMERA_ANGLE,
            "servo_result": str(angle_result)
        }), 200

    except Exception as error:
        print("Ошибка запуска трансляции:", error)

        return jsonify({
            "status": "error",
            "message": str(error)
        }), 500


@app.route("/camera/stop", methods=["POST"])
def stop_camera_stream():
    """
    Останавливает передачу кадров.
    """

    stream_enabled.clear()

    # Разблокируем клиентов, ожидающих следующий кадр.
    with frame_condition:
        frame_condition.notify_all()

    print("Получена команда остановки трансляции")

    return jsonify({
        "status": "stopped",
        "message": "Camera stream stopped"
    }), 200


@app.route("/camera/status", methods=["GET"])
def camera_status():
    """
    Возвращает текущее состояние сервера.
    """

    return jsonify({
        "stream_enabled": stream_enabled.is_set(),
        "frame_number": frame_number,
        "camera_angle": CAMERA_ANGLE
    }), 200


@app.route("/camera/stream", methods=["GET"])
def camera_stream():
    """
    Возвращает MJPEG-видеопоток.
    """

    if not stream_enabled.is_set():
        return jsonify({
            "status": "error",
            "message": "Stream is not started. Send POST /camera/start first."
        }), 409

    print("К видеопотоку подключился клиент")

    return Response(
        generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    capture_thread = threading.Thread(
        target=camera_capture_loop,
        daemon=True
    )
    capture_thread.start()

    print(f"HTTP-сервер камеры запускается на порту {SERVER_PORT}")
    print("Ожидание команды POST /camera/start")

    try:
        app.run(
            host=SERVER_HOST,
            port=SERVER_PORT,
            debug=False,
            use_reloader=False,
            threaded=True
        )

    except KeyboardInterrupt:
        print("Программа остановлена пользователем")

    finally:
        print("Остановка камеры")

        stream_enabled.clear()
        program_stopped.set()

        with frame_condition:
            frame_condition.notify_all()

        camera.stop()

        print("Камера остановлена")