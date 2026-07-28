import argparse
import os
import sys
import time
from pathlib import Path

import cv2

# Keep Ultralytics settings inside the project when the user profile is not writable.
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / "Ultralytics"))

from ultralytics import YOLO


MODEL_PATH = "weights/best.pt"
VIDEO_PATH = "recordings/4.mp4"

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45
MIN_BOX_WIDTH = 20
MIN_BOX_HEIGHT = 20
MIN_BOX_AREA_RATIO = 0.001

CLASS_NAMES = {
    0: "Зарегистрированное",
    1: "Незарегистрированное",
}

ORANGE_COLOR = (0, 165, 255)
CLASS_COLORS = {
    0: (0, 255, 0),
    1: ORANGE_COLOR,
}

WINDOW_NAME = "YOLOv8 Vessel Detection"
MESSAGE_DELAY_SECONDS = 1.0
DEFAULT_FPS = 30.0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Детекция зарегистрированных и незарегистрированных судов на видео YOLOv8."
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Путь к файлу модели .pt")
    parser.add_argument("--video", default=VIDEO_PATH, help="Путь к входному видео")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD, help="Порог confidence")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Порог IoU")
    parser.add_argument(
        "--min-box-width",
        type=int,
        default=MIN_BOX_WIDTH,
        help="Минимальная ширина рамки в пикселях",
    )
    parser.add_argument(
        "--min-box-height",
        type=int,
        default=MIN_BOX_HEIGHT,
        help="Минимальная высота рамки в пикселях",
    )
    parser.add_argument(
        "--min-box-area-ratio",
        type=float,
        default=MIN_BOX_AREA_RATIO,
        help="Минимальная площадь рамки как доля площади кадра",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Не показывать окно OpenCV, обработка продолжится без отображения",
    )
    return parser.parse_args()


def load_model(model_path):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Путь к модели не является файлом: {path}")

    print(f"Загрузка модели: {path}")
    model = YOLO(str(path))
    check_model_classes(model)
    return model


def check_model_classes(model):
    model_names = getattr(model, "names", None)
    if not isinstance(model_names, dict):
        print("Предупреждение: не удалось прочитать model.names из модели.")
        return

    print(f"Классы в модели: {model_names}")
    configured_ids = set(CLASS_NAMES)
    model_ids = set(int(class_id) for class_id in model_names)

    if model_ids != configured_ids:
        print(
            "Предупреждение: набор class_id в model.names отличается от CLASS_NAMES. "
            "Неизвестные классы будут подписаны как 'Неизвестный класс <class_id>'."
        )
        return

    expected_names = set(CLASS_NAMES.values())
    model_name_values = {str(name) for name in model_names.values()}
    if not expected_names.intersection(model_name_values):
        print(
            "Предупреждение: имена классов в model.names не совпадают с русскими "
            "подписями CLASS_NAMES. Будет использоваться отдельная таблица CLASS_NAMES."
        )


def open_video(video_path):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Видеофайл не найден: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Путь к видео не является файлом: {path}")

    print(f"Открытие видео: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Не удалось получить корректный размер видео.")

    if fps <= 0:
        fps = DEFAULT_FPS

    print(f"Размер видео: {width}x{height}")
    print(f"FPS видео: {fps:.2f}")
    return cap, width, height, fps


def clamp_box(x1, y1, x2, y2, frame_width, frame_height):
    x1 = max(0, min(int(round(x1)), frame_width - 1))
    y1 = max(0, min(int(round(y1)), frame_height - 1))
    x2 = max(0, min(int(round(x2)), frame_width - 1))
    y2 = max(0, min(int(round(y2)), frame_height - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def is_box_size_allowed(
    x1,
    y1,
    x2,
    y2,
    frame_width,
    frame_height,
    min_box_width,
    min_box_height,
    min_box_area_ratio,
):
    box_width = x2 - x1
    box_height = y2 - y1
    box_area = box_width * box_height
    min_area = frame_width * frame_height * min_box_area_ratio

    return (
        box_width >= min_box_width
        and box_height >= min_box_height
        and box_area >= min_area
    )


def get_text_color(background_color):
    blue, green, red = background_color
    brightness = 0.114 * blue + 0.587 * green + 0.299 * red
    return (0, 0, 0) if brightness > 140 else (255, 255, 255)


def draw_detection(frame, x1, y1, x2, y2, class_id, confidence):
    class_name = CLASS_NAMES.get(class_id, f"Неизвестный класс {class_id}")
    color = CLASS_COLORS.get(class_id, (255, 255, 255))
    label = f"{class_name} {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size

    frame_height, frame_width = frame.shape[:2]
    label_x1 = x1
    label_y2 = y1 - 6
    if label_y2 - text_height - baseline < 0:
        label_y2 = min(y2 + text_height + baseline + 8, frame_height - 1)

    label_x2 = min(label_x1 + text_width + 8, frame_width - 1)
    label_y1 = max(0, label_y2 - text_height - baseline - 6)

    if label_x2 - label_x1 < text_width + 8:
        label_x1 = max(0, frame_width - text_width - 9)
        label_x2 = frame_width - 1

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
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


def print_detection_message(class_id, confidence, last_message_times):
    now = time.time()
    last_time = last_message_times.get(class_id, 0.0)
    if now - last_time < MESSAGE_DELAY_SECONDS:
        return

    class_name = CLASS_NAMES.get(class_id, f"Неизвестный класс {class_id}")
    lower_name = class_name.lower()
    if class_id in CLASS_NAMES:
        print(f"Обнаружено: {lower_name} судно, уверенность: {confidence:.2f}")
    else:
        print(f"Обнаружено: {lower_name}, уверенность: {confidence:.2f}")
    last_message_times[class_id] = now


def process_detections(
    frame,
    result,
    last_message_times,
    min_box_width,
    min_box_height,
    min_box_area_ratio,
):
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0

    frame_height, frame_width = frame.shape[:2]
    detections_count = 0

    for box in boxes:
        xyxy = box.xyxy[0].detach().cpu().tolist()
        class_id = int(box.cls[0].detach().cpu().item())
        confidence = float(box.conf[0].detach().cpu().item())

        clamped_box = clamp_box(*xyxy, frame_width, frame_height)
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
            min_box_width,
            min_box_height,
            min_box_area_ratio,
        ):
            continue

        draw_detection(frame, x1, y1, x2, y2, class_id, confidence)
        print_detection_message(class_id, confidence, last_message_times)
        detections_count += 1

    return detections_count


def draw_frame_statistics(frame, processing_fps, detections_count):
    lines = (
        f"FPS: {processing_fps:.1f}",
        f"Обнаружено объектов: {detections_count}",
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    padding = 8
    line_height = 26

    max_width = 0
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, text_size[0])

    panel_width = max_width + padding * 2
    panel_height = line_height * len(lines) + padding

    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_width, 8 + panel_height), (0, 0, 0), -1)
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


def calculate_wait_delay_ms(source_fps, frame_started_at):
    target_frame_time = 1.0 / source_fps if source_fps > 0 else 1.0 / DEFAULT_FPS
    elapsed = time.time() - frame_started_at
    remaining = target_frame_time - elapsed
    if remaining <= 0:
        return 1
    return max(1, int(remaining * 1000))


def main():
    args = parse_args()
    cap = None
    processed_frames = 0
    total_processing_time = 0.0
    last_message_times = {}

    try:
        model = load_model(args.model)
        cap, width, height, source_fps = open_video(args.video)

        print("Начало обработки.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_started_at = time.time()

            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                verbose=False,
            )[0]

            detections_count = process_detections(
                frame,
                result,
                last_message_times,
                args.min_box_width,
                args.min_box_height,
                args.min_box_area_ratio,
            )

            processing_time = time.time() - frame_started_at
            total_processing_time += processing_time
            processed_frames += 1
            processing_fps = 1.0 / processing_time if processing_time > 0 else 0.0

            draw_frame_statistics(frame, processing_fps, detections_count)

            if processed_frames % 30 == 0:
                print(
                    f"Кадр: {processed_frames} | "
                    f"FPS обработки: {processing_fps:.1f} | "
                    f"Объектов: {detections_count}"
                )

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, frame)
                delay_ms = calculate_wait_delay_ms(source_fps, frame_started_at)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break

        average_fps = (
            processed_frames / total_processing_time if total_processing_time > 0 else 0.0
        )
        print("Обработка завершена.")
        print(f"Обработано кадров: {processed_frames}")
        print(f"Средний FPS обработки: {average_fps:.1f}")

    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
