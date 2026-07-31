from pioneer_sdk2 import Pioneer              # импортируем класс Pioneer из библиотеки pioneer_sdk2
import pioneer_sdk2                           # импортируем библиотеку pioneer_sdk2 для доступа к событиям
import threading                              # библиотека threading нужна для работы с событием Event
import time                                   # библиотека time содержит функции для работы со временем


drone = Pioneer()                             # создаем экземпляр класса Pioneer, устанавливаем соединение
point_event = threading.Event()               # создаем событие, которое сработает при достижении точки

def point_reached(event):                     # функция вызывается, когда дрон достигает заданной точки
    point_event.set()                         # сообщаем программе, что точка достигнута

def wait_for_point():                         # функция ожидания прилета дрона в точку
    point_event.wait()                        # ждем, пока дрон достигнет заданной точки
    point_event.clear()                       # очищаем событие для следующего ожидания

drone.subscribe(point_reached, pioneer_sdk2.Event.POINT_REACHED) # подписываемся на событие достижения точки

try:                                          # основной код находится внутри блока try
    drone.arm()                               # включаем двигатели
    drone.takeoff()                           # взлетаем

    time.sleep(3)                             # ставим паузу на 3 секунды после взлета

    drone.go_to_local_point(x=0, y=0, z=1, yaw=0, time=3) # летим в точку с координатами x=0, y=0, z=1
                                                          # x, y, z - координаты точки в метрах
                                                          # yaw - поворот по курсу в радианах
                                                          # time - время, за которое требуется достигнуть точку
    wait_for_point()                                      # ждем, пока дрон долетит до заданной точки

    drone.go_to_local_point(x=0, y=1, z=1, yaw=0, time=3) # летим в первую точку с координатами x=0, y=1, z=1
    wait_for_point()                                      # ждем, пока дрон долетит до заданной точки

    drone.go_to_local_point(x=1, y=1, z=1, yaw=0, time=3) # летим во вторую точку с координатами x=1, y=1, z=1
    wait_for_point()                                      # ждем, пока дрон долетит до заданной точки

    drone.land()                              # сажаем дрон, двигатели выключатся автоматически

except KeyboardInterrupt:                     # если пользователь остановил программу сочетанием Ctrl+C
    print("Остановка программы, производится посадка") # выводим сообщение об остановке программы
    drone.land()                                       # сажаем дрон

except Exception as error:                    # если произошла любая другая ошибка
    print("Ошибка:", error)                   # выводим текст ошибки
    drone.land()                              # сажаем дрон при ошибке

finally:                                      # блок finally выполнится в любом случае
    drone.close_connection()                  # закрываем соединение с дроном