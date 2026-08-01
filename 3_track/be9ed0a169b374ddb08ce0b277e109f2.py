import requests
import time

URL = "http://127.0.0.1:5001"

# Тестовые данные
stations = [
    {"x": 0.50, "y": -4.20, "status": "исправна"},
    {"x": 2.20, "y": -2.50, "status": "неисправна"},
    {"x": 3.80, "y": -1.00, "status": "исправна"},
    {"x": 1.10, "y":  0.50, "status": "неисправна"}
]

print(f"Запуск HTTP-симуляции дрона. Отправка на {URL}/drone...")

for i, st in enumerate(stations, 1):
    try:
        # Отправляем словарь напрямую как JSON
        response = requests.post(f"{URL}/drone", json=st, timeout=2)
        print(f"[ОТПРАВЛЕНО {i}/{len(stations)}] {st} | Код ответа: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[ОШИБКА СВЯЗИ] {e}")
    
    time.sleep(1.5) 

print("Симуляция завершена.")