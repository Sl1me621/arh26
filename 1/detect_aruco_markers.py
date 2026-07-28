import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


VIDEO_PATH = "aruco/aruco_flight.mp4"
ARUCO_DICTIONARY = "DICT_4X4_50"
WINDOW_NAME = "ArUco Marker Detection"
DEFAULT_FPS = 30.0
PREPROCESS_MODE = "clahe"
MIN_CONFIRMATION_FRAMES = 3

MARKER_COLOR = (0, 255, 255)
TEXT_COLOR = (255, 255, 255)
TEXT_BACKGROUND_COLOR = (0, 0, 0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Поиск ArUco-меток на видео с выводом ID на экран и в терминал."
    )
    parser.add_argument(
        "--video",
        default=VIDEO_PATH,
        help="Путь к видеофайлу",
    )
    parser.add_argument(
        "--dictionary",
        default=ARUCO_DICTIONARY,
        help="Название словаря ArUco OpenCV, например DICT_4X4_50",
    )
    parser.add_argument(
        "--preprocess",
        choices=("auto", "gray", "clahe", "equalize", "sharpen", "threshold"),
        default=PREPROCESS_MODE,
        help="Предобработка кадра перед поиском меток. auto пробует несколько вариантов",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Не показывать окно OpenCV, только обработать видео и вывести отчёт",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Максимум кадров для обработки. 0 означает обработать всё видео",
    )
    parser.add_argument(
        "--min-confirmation-frames",
        type=int,
        default=MIN_CONFIRMATION_FRAMES,
        help="Сколько кадров должна встретиться метка, чтобы попасть в итоговый список",
    )
    return parser.parse_args()


def get_aruco_detector(dictionary_name):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "Модуль cv2.aruco недоступен. Установите opencv-contrib-python."
        )

    if not hasattr(cv2.aruco, dictionary_name):
        available = ", ".join(
            name for name in dir(cv2.aruco) if name.startswith("DICT_")
        )
        raise ValueError(
            f"Неизвестный словарь ArUco: {dictionary_name}. Доступные: {available}"
        )

    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, dictionary_name)
    )

    if hasattr(cv2.aruco, "DetectorParameters"):
        parameters = cv2.aruco.DetectorParameters()
    else:
        parameters = cv2.aruco.DetectorParameters_create()

    tune_detector_parameters(parameters)

    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, parameters)

    return dictionary, parameters


def tune_detector_parameters(parameters):
    values = {
        "adaptiveThreshWinSizeMin": 3,
        "adaptiveThreshWinSizeMax": 53,
        "adaptiveThreshWinSizeStep": 4,
        "adaptiveThreshConstant": 7,
        "minMarkerPerimeterRate": 0.01,
        "maxMarkerPerimeterRate": 4.0,
        "polygonalApproxAccuracyRate": 0.05,
        "minCornerDistanceRate": 0.03,
        "minDistanceToBorder": 2,
        "errorCorrectionRate": 0.6,
    }

    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        values["cornerRefinementMethod"] = cv2.aruco.CORNER_REFINE_SUBPIX

    for name, value in values.items():
        if hasattr(parameters, name):
            setattr(parameters, name, value)


def open_video(video_path):
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Видеофайл не найден: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Путь к видео не является файлом: {path}")

    print(f"Открытие видео: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Не удалось получить корректный размер видео.")

    if fps <= 0:
        fps = DEFAULT_FPS

    print(f"Размер видео: {width}x{height}")
    print(f"FPS видео: {fps:.2f}")
    return capture, fps


def detect_markers_on_image(image, detector):
    if hasattr(detector, "detectMarkers"):
        corners, ids, rejected = detector.detectMarkers(image)
    else:
        dictionary, parameters = detector
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image,
            dictionary,
            parameters=parameters,
        )

    return corners, ids


def build_preprocessed_images(frame, preprocess_mode):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    clahe_image = clahe.apply(gray)
    equalized = cv2.equalizeHist(gray)
    thresholded = cv2.adaptiveThreshold(
        clahe_image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )

    images = {
        "gray": gray,
        "clahe": clahe_image,
        "equalize": equalized,
        "sharpen": sharpened,
        "threshold": thresholded,
    }

    if preprocess_mode == "auto":
        return [
            ("gray", gray),
            ("clahe", clahe_image),
            ("equalize", equalized),
            ("sharpen", sharpened),
            ("threshold", thresholded),
        ]

    return [(preprocess_mode, images[preprocess_mode])]


def detect_aruco_markers(frame, detector, preprocess_mode):
    found_by_id = {}

    for preprocess_name, image in build_preprocessed_images(frame, preprocess_mode):
        corners, ids = detect_markers_on_image(image, detector)

        if ids is None or len(ids) == 0:
            continue

        for marker_id, marker_corners in zip(ids.flatten(), corners):
            marker_id = int(marker_id)
            if marker_id in found_by_id:
                continue

            normalized_corners = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
            found_by_id[marker_id] = {
                "id": marker_id,
                "corners": normalized_corners,
                "center": get_marker_center(normalized_corners),
                "preprocess": preprocess_name,
            }

    return [found_by_id[marker_id] for marker_id in sorted(found_by_id)]


def get_marker_center(corners):
    center_x = float(np.mean(corners[:, 0]))
    center_y = float(np.mean(corners[:, 1]))
    return int(round(center_x)), int(round(center_y))


def draw_text_with_background(frame, text, origin, font_scale=0.7, thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    frame_height, frame_width = frame.shape[:2]

    x = max(0, min(origin[0], frame_width - text_width - 10))
    y = max(text_height + 8, min(origin[1], frame_height - baseline - 4))

    background_x1 = x - 4
    background_y1 = y - text_height - baseline - 4
    background_x2 = x + text_width + 4
    background_y2 = y + baseline + 4

    cv2.rectangle(
        frame,
        (background_x1, background_y1),
        (background_x2, background_y2),
        TEXT_BACKGROUND_COLOR,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def draw_marker(frame, marker):
    corners = marker["corners"].astype(np.int32)
    center = marker["center"]
    marker_id = marker["id"]

    cv2.polylines(frame, [corners], True, MARKER_COLOR, 2)
    cv2.circle(frame, center, 4, MARKER_COLOR, -1)

    label_x = int(corners[:, 0].min())
    label_y = int(corners[:, 1].min()) - 8
    draw_text_with_background(frame, f"ID: {marker_id}", (label_x, label_y))


def draw_frame_statistics(frame, frame_number, current_ids, confirmed_ids, processing_fps):
    current_ids_text = ", ".join(str(marker_id) for marker_id in current_ids)
    if not current_ids_text:
        current_ids_text = "-"

    confirmed_ids_text = ", ".join(str(marker_id) for marker_id in sorted(confirmed_ids))
    if not confirmed_ids_text:
        confirmed_ids_text = "-"

    lines = [
        f"Кадр: {frame_number}",
        f"FPS: {processing_fps:.1f}",
        f"Метки на кадре: {current_ids_text}",
        f"Подтверждено меток: {len(confirmed_ids)}",
        f"ID: {confirmed_ids_text}",
    ]

    y = 28
    for line in lines:
        draw_text_with_background(frame, line, (12, y), font_scale=0.65, thickness=2)
        y += 30


def update_marker_confirmation(current_ids, marker_counts, confirmed_ids, min_frames):
    newly_confirmed_ids = []
    newly_seen_ids = []

    for marker_id in current_ids:
        if marker_id not in marker_counts:
            newly_seen_ids.append(marker_id)

        marker_counts[marker_id] = marker_counts.get(marker_id, 0) + 1
        if marker_counts[marker_id] < min_frames or marker_id in confirmed_ids:
            continue

        confirmed_ids.add(marker_id)
        newly_confirmed_ids.append(marker_id)

    return newly_seen_ids, newly_confirmed_ids


def print_new_marker_messages(newly_seen_ids):
    for marker_id in newly_seen_ids:
        print(f"Обнаружена ArUco-метка, ID: {marker_id}")


def print_confirmed_marker_messages(newly_confirmed_ids, marker_counts):
    for marker_id in newly_confirmed_ids:
        print(
            f"Подтверждена ArUco-метка, ID: {marker_id}, "
            f"обнаружений: {marker_counts[marker_id]}"
        )


def calculate_wait_delay_ms(source_fps, frame_started_at):
    target_frame_time = 1.0 / source_fps if source_fps > 0 else 1.0 / DEFAULT_FPS
    elapsed = time.time() - frame_started_at
    remaining = target_frame_time - elapsed
    if remaining <= 0:
        return 1
    return max(1, int(remaining * 1000))


def print_final_report(processed_frames, confirmed_ids, marker_counts, total_processing_time):
    average_fps = (
        processed_frames / total_processing_time if total_processing_time > 0 else 0.0
    )

    print()
    print("Обработка завершена.")
    print(f"Обработано кадров: {processed_frames}")
    print(f"Средний FPS обработки: {average_fps:.1f}")
    print(f"Количество уникальных подтверждённых меток: {len(confirmed_ids)}")

    if confirmed_ids:
        ids_text = ", ".join(str(marker_id) for marker_id in sorted(confirmed_ids))
        print(f"ID уникальных подтверждённых меток: {ids_text}")
    else:
        print("ID уникальных подтверждённых меток: нет")

    unconfirmed_ids = sorted(set(marker_counts) - set(confirmed_ids))
    if unconfirmed_ids:
        details = ", ".join(
            f"{marker_id} ({marker_counts[marker_id]})"
            for marker_id in unconfirmed_ids
        )
        print(f"Неподтверждённые кандидаты: {details}")


def main():
    args = parse_args()
    capture = None
    processed_frames = 0
    total_processing_time = 0.0
    confirmed_ids = set()
    marker_counts = {}

    try:
        if args.min_confirmation_frames < 1:
            raise ValueError("--min-confirmation-frames должен быть больше или равен 1")

        print(f"Создание ArUco-детектора: {args.dictionary}")
        print(f"Предобработка: {args.preprocess}")
        print(f"Кадров для подтверждения: {args.min_confirmation_frames}")
        detector = get_aruco_detector(args.dictionary)
        capture, source_fps = open_video(args.video)
        print("Начало обработки.")

        while True:
            if args.max_frames > 0 and processed_frames >= args.max_frames:
                break

            ok, frame = capture.read()
            if not ok:
                break

            frame_started_at = time.time()
            markers = detect_aruco_markers(frame, detector, args.preprocess)
            current_ids = sorted({marker["id"] for marker in markers})
            newly_seen_ids, newly_confirmed_ids = update_marker_confirmation(
                current_ids,
                marker_counts,
                confirmed_ids,
                args.min_confirmation_frames,
            )

            for marker in markers:
                draw_marker(frame, marker)

            print_new_marker_messages(newly_seen_ids)
            print_confirmed_marker_messages(newly_confirmed_ids, marker_counts)

            processing_time = time.time() - frame_started_at
            total_processing_time += processing_time
            processed_frames += 1
            processing_fps = 1.0 / processing_time if processing_time > 0 else 0.0

            draw_frame_statistics(
                frame,
                processed_frames,
                current_ids,
                confirmed_ids,
                processing_fps,
            )

            if processed_frames % 30 == 0:
                ids_text = ", ".join(str(marker_id) for marker_id in current_ids)
                if not ids_text:
                    ids_text = "-"
                print(
                    f"Кадр: {processed_frames} | "
                    f"FPS обработки: {processing_fps:.1f} | "
                    f"Метки: {ids_text} | "
                    f"Подтверждено: {len(confirmed_ids)}"
                )

            if not args.no_display:
                cv2.imshow(WINDOW_NAME, frame)
                delay_ms = calculate_wait_delay_ms(source_fps, frame_started_at)
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    break

        print_final_report(
            processed_frames,
            confirmed_ids,
            marker_counts,
            total_processing_time,
        )

    except Exception as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        sys.exit(1)

    finally:
        if capture is not None:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
