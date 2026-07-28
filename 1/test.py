import time

import cv2
import numpy as np

from pioneer_rknn import Yolo
from pioneer_sdk2 import Camera, CameraType, ImageViewer, ServoCamera


# ============================================================
# НАСТРОЙКИ
# ============================================================

MODEL_NAME = "qwert"

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

OBJECT_THRESHOLD = 0.60
NMS_THRESHOLD = 0.45

STREAM_NAME = "boats_test"
STREAM_FPS = 10
SERVO_ANGLE = -80

# Фильтрация слишком маленьких рамок.
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20
MIN_BOX_AREA_RATIO = 0.001

# Не выводить одинаковое сообщение чаще одного раза в этот интервал.
MESSAGE_DELAY_SECONDS = 1.0

# Названия для терминала.
TERMINAL_CLASS_NAMES = {
    0: "зарегистрированное",
    1: "незарегистрированное",
}

# OpenCV FONT_HERSHEY_SIMPLEX обычно не поддерживает кириллицу,
# поэтому для видеопотока используются латинские подписи.
VIDEO_CLASS_NAMES = {
    0: "registered",
    1: "unregistered",
}

# Цвета OpenCV в формате BGR.
CLASS_COLORS = {
    0: (0, 255, 0),      # зелёный
    1: (0, 165, 255),    # оранжевый
}


# ============================================================
# ПОДГОТОВКА ВХОДА RKNN
# ============================================================


def prepare_input(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает кадр для показа и вход RKNN: RGB NHWC uint8 с batch."""

    display_frame = cv2.resize(
        frame,
        (INPUT_WIDTH, INPUT_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )

    rgb_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    # (640, 640, 3) -> (1, 640, 640, 3)
    input_tensor = np.expand_dims(rgb_image, axis=0)
    input_tensor = np.ascontiguousarray(input_tensor, dtype=np.uint8)

    return display_frame, input_tensor


# ============================================================
# ОБРАБОТКА РЕЗУЛЬТАТА YOLO
# ============================================================


def unpack_detections(result):
    """Нормализует результат Yolo.run() в три последовательности."""

    if result is None:
        return [], [], []

    if not isinstance(result, (tuple, list)) or len(result) != 3:
        print(
            "Предупреждение: неожиданный формат результата YOLO:",
            type(result),
            repr(result),
        )
        return [], [], []

    boxes, classes, scores = result

    if boxes is None or classes is None or scores is None:
        return [], [], []

    return boxes, classes, scores


def clamp_box(box, frame_width: int, frame_height: int):
    """Ограничивает рамку границами кадра."""

    values = np.asarray(box, dtype=np.float32).reshape(-1)
    if values.size < 4:
        return None

    x1, y1, x2, y2 = values[:4]

    x1 = max(0, min(int(round(float(x1))), frame_width - 1))
    y1 = max(0, min(int(round(float(y1))), frame_height - 1))
    x2 = max(0, min(int(round(float(x2))), frame_width - 1))
    y2 = max(0, min(int(round(float(y2))), frame_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def is_box_size_allowed(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    frame_width: int,
    frame_height: int,
) -> bool:
    """Отбрасывает слишком маленькие детекции."""

    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    min_area = frame_width * frame_height * MIN_BOX_AREA_RATIO

    return (
        box_width >= MIN_BOX_WIDTH
        and box_height >= MIN_BOX_HEIGHT
        and box_area >= min_area
    )


def get_text_color(background_color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Выбирает чёрный или белый текст по яркости фона."""

    blue, green, red = background_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness > 140 else (255, 255, 255)


def draw_detection(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    class_id: int,
    confidence: float,
) -> None:
    """Рисует рамку и цветную плашку класса."""

    class_name = VIDEO_CLASS_NAMES.get(class_id, f"class_{class_id}")
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    label = f"{class_name} {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    frame_height, frame_width = frame.shape[:2]

    label_x1 = x1
    label_y2 = y1 - 6

    # Если над рамкой нет места, переносим подпись внутрь/под верхнюю границу.
    if label_y2 - text_height - baseline < 0:
        label_y2 = min(y1 + text_height + baseline + 10, frame_height - 1)

    label_x2 = min(label_x1 + text_width + 8, frame_width - 1)
    label_y1 = max(0, label_y2 - text_height - baseline - 6)

    if label_x2 - label_x1 < text_width + 8:
        label_x1 = max(0, frame_width - text_width - 9)
        label_x2 = frame_width - 1

    cv2.rectangle(
        frame,
        (label_x1, label_y1),
        (label_x2, label_y2),
        color,
        -1,
    )

    cv2.putText(
        frame,
        label,
        (label_x1 + 4, label_y2 - baseline - 3),
        font,
        font_scale,
        get_text_color(color),
        thickness,
        cv2.LINE_AA,
    )


def print_detection_message(
    class_id: int,
    confidence: float,
    last_message_times: dict[int, float],
) -> None:
    """Печатает обнаружение, но не чаще заданного интервала для класса."""

    now = time.monotonic()
    last_time = last_message_times.get(class_id, 0.0)

    if now - last_time < MESSAGE_DELAY_SECONDS:
        return

    class_name = TERMINAL_CLASS_NAMES.get(class_id, f"неизвестный класс {class_id}")

    if class_id in TERMINAL_CLASS_NAMES:
        print(
            f"Обнаружено: {class_name} судно, "
            f"уверенность: {confidence:.2f}"
        )
    else:
        print(
            f"Обнаружено: {class_name}, "
            f"уверенность: {confidence:.2f}"
        )

    last_message_times[class_id] = now


def process_detections(
    frame: np.ndarray,
    result,
    last_message_times: dict[int, float],
) -> int:
    """Фильтрует, рисует и выводит обнаружения. Возвращает их количество."""

    boxes, classes, scores = unpack_detections(result)

    frame_height, frame_width = frame.shape[:2]
    detections_count = 0

    for box, class_id, confidence in zip(boxes, classes, scores):
        class_id = int(class_id)
        confidence = float(confidence)

        clamped_box = clamp_box(box, frame_width, frame_height)
        if clamped_box is None:
            continue

        x1, y1, x2, y2 = clamped_box

        if not is_box_size_allowed(
            x1,
            y1,
            x2,
            y2,
            frame_width,
            frame_height,
        ):
            continue

        draw_detection(
            frame,
            x1,
            y1,
            x2,
            y2,
            class_id,
            confidence,
        )

        print_detection_message(
            class_id,
            confidence,
            last_message_times,
        )

        detections_count += 1

    return detections_count


# ============================================================
# СТАТИСТИКА НА КАДРЕ
# ============================================================


def draw_frame_statistics(
    frame: np.ndarray,
    inference_fps: float,
    detections_count: int,
    inference_ms: float,
) -> None:
    """Рисует полупрозрачную панель статистики."""

    # Латиница используется для гарантированной поддержки cv2.putText.
    lines = (
        f"FPS: {inference_fps:.1f}",
        f"Objects: {detections_count}",
        f"Inference: {inference_ms:.1f} ms",
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    padding = 8
    line_height = 25

    max_width = 0
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_size[0])

    panel_width = max_width + padding * 2
    panel_height = line_height * len(lines) + padding

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (8, 8),
        (8 + panel_width, 8 + panel_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    y = 8 + padding + 16
    for line in lines:
        cv2.putText(
            frame,
            line,
            (8 + padding, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height


# ============================================================
# ОСНОВНАЯ ПРОГРАММА
# ============================================================


def main() -> None:
    camera = None
    viewer = None
    model = None
    servo_camera = None

    frame_number = 0
    last_message_times: dict[int, float] = {}

    try:
        print("Инициализация сервопривода камеры...")
        servo_camera = ServoCamera()
        servo_result = servo_camera.set_angle(SERVO_ANGLE)
        print(f"Угол камеры установлен: {SERVO_ANGLE}, результат: {servo_result}")

        # Даём механизму немного времени занять положение.
        time.sleep(0.5)

        print("Запуск камеры...")
        camera = Camera(camera_type=CameraType.MAIN)

        viewer = ImageViewer()

        print("Загрузка RKNN-модели...")
        model = Yolo(
            model_name=MODEL_NAME,
            object_thresh=OBJECT_THRESHOLD,
            nms_thresh=NMS_THRESHOLD,
            img_width=INPUT_WIDTH,
            img_height=INPUT_HEIGHT,
        )

        print("Модель загружена.")
        print("Модель:", MODEL_NAME)
        print("Порог confidence:", OBJECT_THRESHOLD)
        print("Классы:", TERMINAL_CLASS_NAMES)
        print("Открой поток:")
        print(f"http://10.42.0.1:8889/{STREAM_NAME}/")

        first_frame = True

        while True:
            frame = camera.get_cv_frame(timeout=2.0)

            if frame is None:
                print("Кадр с камеры не получен")
                continue

            frame_number += 1
            display_frame, input_tensor = prepare_input(frame)

            if first_frame:
                print("Исходный кадр:", frame.shape, frame.dtype)
                print(
                    "Вход RKNN:",
                    input_tensor.shape,
                    input_tensor.dtype,
                    "range:",
                    int(input_tensor.min()),
                    "...",
                    int(input_tensor.max()),
                )
                first_frame = False

            inference_started_at = time.perf_counter()
            result = model.run([input_tensor])
            inference_time = time.perf_counter() - inference_started_at

            inference_fps = 1.0 / inference_time if inference_time > 0 else 0.0
            inference_ms = inference_time * 1000.0

            detections_count = process_detections(
                display_frame,
                result,
                last_message_times,
            )

            draw_frame_statistics(
                display_frame,
                inference_fps,
                detections_count,
                inference_ms,
            )

            if frame_number % 10 == 0:
                print(
                    f"Кадр {frame_number} | "
                    f"объектов: {detections_count} | "
                    f"инференс: {inference_ms:.1f} мс | "
                    f"FPS: {inference_fps:.1f}"
                )

            viewer.imshow(
                name=STREAM_NAME,
                frame=display_frame,
                fps=STREAM_FPS,
            )

    except KeyboardInterrupt:
        print("\nОстановлено пользователем")

    except Exception as error:
        print(f"\nОшибка: {type(error).__name__}: {error}")
        raise

    finally:
        print("Освобождение ресурсов...")

        if model is not None:
            try:
                model.release()
            except Exception as error:
                print("Ошибка освобождения модели:", error)

        if camera is not None:
            try:
                camera.stop()
            except Exception as error:
                print("Ошибка остановки камеры:", error)

        if viewer is not None:
            try:
                viewer.close()
            except Exception as error:
                print("Ошибка закрытия потока:", error)

        if servo_camera is not None and hasattr(servo_camera, "close"):
            try:
                servo_camera.close()
            except Exception as error:
                print("Ошибка закрытия ServoCamera:", error)

        print("Готово")


if __name__ == "__main__":
    main()