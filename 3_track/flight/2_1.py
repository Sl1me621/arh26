from pioneer_sdk2 import Pioneer
import pioneer_sdk2

import threading
import time

from flask import Flask, jsonify


SERVER_HOST = "0.0.0.0"
SERVER_PORT = 5001

z=0.8

ROUTE    = [
    (0.5, 2.0, z, 0.0, 0),
    (0.5, -0.5, z, 0.0, 0),
    (-0.5, -0.5, z, 0.0, 0),
    (-0.5, 2.0, z, 0.0, 0)    ]



point_event = threading.Event()
start_event = threading.Event()

state_lock = threading.Lock()
mission_started = False


app = Flask(__name__)


@app.route("/start", methods=["POST"])
def receive_start_command():

    with state_lock:
        if mission_started or start_event.is_set():
            return jsonify({
                "status": "error",
                "message": "mission already started"
            }), 409

        start_event.set()

    print("Получена команда на запуск миссии")

    return jsonify({
        "status": "accepted",
        "message": "start command received"
    }), 202


@app.route("/status", methods=["GET"])
def get_status():

    with state_lock:
        started = mission_started
        command_received = start_event.is_set()

    return jsonify({
        "mission_started": started,
        "start_command_received": command_received
    }), 200


def run_http_server():

    app.run(
        host=SERVER_HOST,
        port=SERVER_PORT,
        debug=False,
        use_reloader=False,
        threaded=True
    )


def point_reached(event):

    point_event.set()


def wait_for_point(timeout: float = 30.0):

    reached = point_event.wait(timeout=timeout)

    if not reached:
        raise TimeoutError(
            f"Дрон не достиг точки за {timeout} секунд"
        )

    point_event.clear()


def fly_to_point(
    drone: Pioneer,
    x: float,
    y: float,
    z: float,
    yaw: float,
    duration: int
):

    # Удаляем возможное событие от предыдущей точки.
    point_event.clear()

    print(
        f"Летим в точку: "
        f"x={x}, y={y}, z={z}, "
        f"yaw={yaw}, time={duration}"
    )

    command_sent = drone.go_to_local_point(
        x=x,
        y=y,
        z=z,
        yaw=yaw,
        time=duration
    )

    if command_sent is False:
        raise RuntimeError(
            f"Команда перехода в точку x={x}, y={y}, z={z} отклонена"
        )

    # Дополнительный запас времени на стабилизацию и достижение точки.
    wait_for_point(timeout=max(duration + 10, 15))

    print("Точка достигнута")


# ---------------------------------------------------------------------
# Основная программа
# ---------------------------------------------------------------------

drone = None
airborne = False

try:
    print("Подключение к автопилоту")

    drone = Pioneer()

    drone.subscribe(
        point_reached,
        pioneer_sdk2.Event.POINT_REACHED
    )

    # HTTP работает в фоновом потоке.
    server_thread = threading.Thread(
        target=run_http_server,
        daemon=True
    )
    server_thread.start()

    print(
        f"HTTP-сервер запущен на порту {SERVER_PORT}"
    )
    print("Ожидание команды POST /start")

    # Основной поток останавливается здесь до получения HTTP-команды.
    start_event.wait()

    with state_lock:
        mission_started = True
        start_event.clear()

    print("Команда принята. Начинается полёт")

    # Включаем двигатели.
    if drone.arm() is False:
        raise RuntimeError("Не удалось включить двигатели")
    time.sleep(1)
    print("Двигатели включены")

    # Выполняем взлёт.
    if drone.takeoff() is False:
        raise RuntimeError("Не удалось выполнить взлёт")

    airborne = True

    print("Взлёт завершён")

    time.sleep(3)

    # Последовательно проходим маршрут.
    for point_number, point in enumerate(ROUTE, start=1):
        print(f"Точка маршрута №{point_number}")

        fly_to_point(
            drone=drone,
            x=point[0],
            y=point[1],
            z=point[2],
            yaw=point[3],
            duration=point[4]
        )
    fly_to_point(
        drone=drone,
        x=0.0,
        y=0.0,
        z=0.8,
        yaw=0.0,
        duration=0
    )

    time.sleep(3)    
    # Возвращение над местом старта.
    fly_to_point(
        drone=drone,
        x=0.0,
        y=0.0,
        z=0.3,
        yaw=0.0,
        duration=0
    )    
    time.sleep(3)   
    print("Маршрут завершён. Выполняется посадка")

    if drone.land() is False:
        raise RuntimeError("Команда посадки была отклонена")

    airborne = False

    print("Посадка завершена")

except KeyboardInterrupt:
    print("Программа остановлена пользователем")

    if drone is not None and airborne:
        print("Выполняется аварийная посадка")

        try:
            drone.land()
            airborne = False
        except Exception as landing_error:
            print("Ошибка аварийной посадки:", landing_error)

except Exception as error:
    print("Ошибка:", error)

    if drone is not None and airborne:
        print("Выполняется посадка из-за ошибки")

        try:
            drone.land()
            airborne = False
        except Exception as landing_error:
            print("Ошибка аварийной посадки:", landing_error)

finally:
    if drone is not None:
        drone.close_connection()
        print("Соединение с дроном закрыто")