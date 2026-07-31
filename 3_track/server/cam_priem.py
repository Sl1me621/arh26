import cv2
import requests


DRONE_URL = "http://172.17.49.101:5001"

START_URL = f"{DRONE_URL}/camera/start"
STREAM_URL = f"{DRONE_URL}/camera/stream"
STOP_URL = f"{DRONE_URL}/camera/stop"

VIDEO_FILE = "flight.mp4"
RECORD_FPS = 30


def start_stream():
    response = requests.post(
        START_URL,
        timeout=10
    )

    print("Запуск трансляции:")
    print("HTTP-код:", response.status_code)

    try:
        print("Ответ:", response.json())
    except ValueError:
        print("Ответ:", response.text)

    response.raise_for_status()


def stop_stream():
    try:
        response = requests.post(
            STOP_URL,
            timeout=5
        )

        print("Остановка трансляции:")
        print("HTTP-код:", response.status_code)

        try:
            print("Ответ:", response.json())
        except ValueError:
            print("Ответ:", response.text)

    except requests.RequestException as error:
        print("Не удалось отправить команду остановки:", error)


def show_and_record_stream():
    print("Подключение к видеопотоку:", STREAM_URL)

    video = cv2.VideoCapture(STREAM_URL)

    if not video.isOpened():
        raise RuntimeError(
            "OpenCV не смог открыть видеопоток. "
            "Проверьте адрес сервера и порт 5001."
        )

    video_writer = None

    print("Трансляция началась")
    print(f"Видео будет сохранено в файл: {VIDEO_FILE}")
    print("Для завершения нажмите Q")

    try:
        while True:
            success, frame = video.read()

            if not success or frame is None:
                print("Не удалось получить кадр")
                break

            # Создаём VideoWriter после получения первого кадра.
            if video_writer is None:
                height, width = frame.shape[:2]

                codec = cv2.VideoWriter_fourcc(*"mp4v")

                video_writer = cv2.VideoWriter(
                    VIDEO_FILE,
                    codec,
                    RECORD_FPS,
                    (width, height)
                )

                if not video_writer.isOpened():
                    raise RuntimeError(
                        "Не удалось создать файл для записи видео"
                    )

                print(
                    f"Запись началась: {width}x{height}, "
                    f"{RECORD_FPS} FPS"
                )

            # Записываем текущий кадр в flight.mp4.
            video_writer.write(frame)

            cv2.imshow(
                "Pioneer Mini 2 camera",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Получена команда завершения")
                break

    finally:
        video.release()

        if video_writer is not None:
            video_writer.release()
            print(f"Видео сохранено: {VIDEO_FILE}")

        cv2.destroyAllWindows()


def main():
    try:
        start_stream()
        show_and_record_stream()

    except requests.ConnectionError:
        print(
            "Не удалось подключиться к Pioneer Mini 2. "
            "Проверьте подключение к Wi-Fi дрона."
        )

    except requests.Timeout:
        print("Pioneer Mini 2 не ответил вовремя")

    except Exception as error:
        print("Ошибка:", error)

    finally:
        stop_stream()


if __name__ == "__main__":
    main()