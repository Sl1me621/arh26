from pioneer_sdk2 import Pioneer, Camera, ImageViewer, ServoCamera
import threading
servo_camera = ServoCamera()
import time                                    # библиотека time содержит функции для работы со временем

drone = Pioneer()                              # создаем экземпляр класса Pioneer, устанавливаем соединение
camera = Camera()                              # создаем экземпляр класса Camera для получения кадров с камеры
viewer = ImageViewer()                         # создаем экземпляр класса ImageViewer для трансляции видео
result = servo_camera.set_angle(-80)
camera_stop_event = threading.Event()


def camera_stream():
    """Постоянно получает кадры и отправляет их в трансляцию."""

    print("Поток камеры запущен")

    while not camera_stop_event.is_set():
        try:
            frame = camera.get_cv_frame(timeout=1.0)

            if frame is not None:
                viewer.imshow(
                    "video",
                    frame,
                    fps=30,
                )

        except Exception as error:
            print("Ошибка камеры:", error)
            time.sleep(0.2)

    print("Поток камеры остановлен")



def wait_for_point(timeout=25):
    deadline = time.monotonic() + timeout
    while not drone.point_reached():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Дрон не достиг точки за {timeout} секунд"
            )

        time.sleep(0.1)

def go_to_start_point():                       # функция взлета и выхода в начальную точку
    drone.arm()    
    if not drone.arm():
        raise RuntimeError("Не удалось запустить моторы")
    # time.sleep(3) 
    drone.takeoff()       
    if not drone.takeoff():
        raise RuntimeError("Не удалось выполнить взлёт")                     # производим взлет
    # time.sleep(3)  # обязательно чтоб успел взлететь

    drone.go_to_local_point(x=0, y=0, z=1.8, yaw=0) # летим в начальную точку

    wait_for_point()                           # ждем, пока дрон долетит до начальной точки


def hover(seconds: float):                     # функция зависания на текущей точке
    end_time = time.time() + seconds          # вычисляем время окончания
    while time.time() < end_time:             # ждем указанное время                        # показываем видео во время зависания
        time.sleep(0.1)                       # пауза чтобы не нагружать процессор


def fly_through_points(points):                # функция полета по заданным точкам
    for point in points:                       # перебираем все точки из списка
        drone.go_to_local_point(               # отправляем дрон в текущую точку
            x=point["x"],                      # координата точки по оси X
            y=point["y"],                      # координата точки по оси Y
            z=point["z"]   ,
            yaw=0 ,
                           # координата точки по оси Z
                                         # время, за которое нужно достигнуть точку
        )

        wait_for_point()                       # ждем, пока дрон долетит до текущей точки
        hover(5)


ALTITUDE = 1.8  # фиксированная высота полета (метры)

# waypoints = [
#     (1.40, 1.75),
#     (2.40, 1.75),
#     (3.40, 1.75),
#     (3.40, 2.75),
#     (2.40, 2.75),
#     (1.40, 2.75)





waypoints = [
    (0, 2),
    (2.5, -0.5),
    (3.0, -0.5),
    (3.5, -0.5),
    (4.0, -0.5),
    (4.5, -0.5),
    (4.5, 0.5),
    (4.0, 0.5),
    (3.5, 0.5),
    (3.0, 0.5),
    (2.5, 0.5)

]



def parse_waypoints(raw):
    result = []
    for x, y in raw:
        result.append({"x": x, "y": y, "z": ALTITUDE, "yaw": 0})
    return result


camera_thread = threading.Thread(
    target=camera_stream,
    daemon=True,
)

camera_thread.start()

try:                                           # основной код находится внутри блока try
    go_to_start_point()                        # выполняем взлет и выход в начальную точку
    fly_through_points(parse_waypoints(waypoints))  # выполняем полет по заданным точкам



    drone.go_to_local_point(               # отправляем дрон на старт
                x=0,                      # координата точки по оси X
                y=0,                      # координата точки по оси Y
                z=ALTITUDE,
                yaw=0                  # координата точки по оси Z
                                             # время, за которое нужно достигнуть точку
            )
    
    wait_for_point()  


    
    drone.land()                               # производим посадку после завершения полета

except KeyboardInterrupt:                      # если пользователь остановил программу сочетанием Ctrl+C
    print("Остановка программы, производится посадка") # выводим сообщение об остановке программы
    drone.land()                                       # сажаем дрон

except Exception as error:                     # если произошла другая ошибка
    print("Ошибка:", error)                    # выводим текст ошибки
    drone.land()                               # сажаем дрон при ошибке

finally: 
    camera_stop_event.set()

    if camera_thread.is_alive():
        camera_thread.join(timeout=2.0)
                                      # блок finally выполнится в любом случае
    viewer.close()                             # останавливаем RTSP-трансляцию
    camera.stop()                              # останавливаем получение кадров с камеры
    drone.close_connection()                   # закрываем соединение с квадрокоптером