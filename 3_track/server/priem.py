import requests


# При прямом подключении компьютера
# к точке доступа Pioneer Mini 2.
DRONE_URL = "http://172.17.49.101:5001"


def send_start_command():
    try:
        response = requests.post(
            f"{DRONE_URL}/start",
            timeout=5
        )

        print("HTTP-код:", response.status_code)

        try:
            print("Ответ дрона:", response.json())
        except ValueError:
            print("Ответ дрона:", response.text)

        if response.status_code == 202:
            print("Команда принята. Дрон начинает выполнение миссии.")

        elif response.status_code == 409:
            print("Миссия уже запущена.")

        else:
            print("Дрон вернул неожиданный ответ.")

    except requests.ConnectionError:
        print(
            "Не удалось подключиться к дрону. "
            "Проверьте Wi-Fi, IP-адрес и запущенный сервер."
        )

    except requests.Timeout:
        print("Дрон не ответил за отведённое время.")

    except requests.RequestException as error:
        print("Ошибка HTTP-запроса:", error)


if __name__ == "__main__":
    send_start_command()