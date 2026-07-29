from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


VIDEO_WINDOW = "Contour tuner"
MASK_WINDOW = "White mask"
CONTROLS_WINDOW = "Threshold controls"
FRAME_TRACKBAR = "Frame"

DEFAULT_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Подбор порогов поиска белого прямоугольного маркера."
    )
    parser.add_argument(
        "video",
        nargs="?",
        help="Путь к видео",
    )
    return parser.parse_args()


def choose_video(video_argument: str | None) -> Path | None:
    """Получает путь из аргумента или открывает окно выбора файла."""

    if video_argument:
        return Path(video_argument)

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        file_name = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[
                (
                    "Video files",
                    "*.mp4 *.avi *.mov *.mkv *.m4v",
                ),
                ("All files", "*.*"),
            ],
        )

        root.destroy()

        if not file_name:
            return None

        return Path(file_name)

    except Exception as error:
        print(
            f"Не удалось открыть окно выбора файла: {error}",
            file=sys.stderr,
        )
        return None


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int]:
    """Открывает видео и возвращает VideoCapture, FPS и число кадров."""

    if not path.is_file():
        raise FileNotFoundError(
            f"Видео не найдено: {path}"
        )

    capture = cv2.VideoCapture(str(path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Не удалось открыть видео: {path}"
        )

    fps = float(
        capture.get(cv2.CAP_PROP_FPS)
    )

    if not math.isfinite(fps) or fps <= 0:
        fps = DEFAULT_FPS

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    return capture, fps, max(0, frame_count)


def empty_callback(_: int) -> None:
    """Пустой обработчик OpenCV trackbar."""


def odd_kernel(value: int) -> int:
    """Преобразует значение ползунка в положительный нечётный размер."""

    value = max(1, value)

    if value % 2 == 0:
        value += 1

    return value


def create_windows(
    frame_count: int,
    seek_state: dict,
) -> None:
    """Создаёт окна и ползунки."""

    cv2.namedWindow(
        VIDEO_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        MASK_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        CONTROLS_WINDOW,
        cv2.WINDOW_NORMAL,
    )

    def on_frame_trackbar(value: int) -> None:
        if not seek_state["updating"]:
            seek_state["frame"] = value

    cv2.createTrackbar(
        FRAME_TRACKBAR,
        VIDEO_WINDOW,
        0,
        max(1, frame_count - 1),
        on_frame_trackbar,
    )

    # HSV-порог белого цвета
    cv2.createTrackbar(
        "S max",
        CONTROLS_WINDOW,
        110,
        255,
        empty_callback,
    )

    cv2.createTrackbar(
        "V min",
        CONTROLS_WINDOW,
        145,
        255,
        empty_callback,
    )

    # Минимальная площадь:
    # значение 80 означает 80 / 10000 = 0.008
    cv2.createTrackbar(
        "Min area x10000",
        CONTROLS_WINDOW,
        80,
        3000,
        empty_callback,
    )

    # Максимальная площадь в процентах
    cv2.createTrackbar(
        "Max area percent",
        CONTROLS_WINDOW,
        70,
        100,
        empty_callback,
    )

    # Значение 165 означает aspect ratio = 1.65
    cv2.createTrackbar(
        "Max aspect x100",
        CONTROLS_WINDOW,
        165,
        400,
        empty_callback,
    )

    # Значение 48 означает rectangularity = 0.48
    cv2.createTrackbar(
        "Min rectangularity",
        CONTROLS_WINDOW,
        48,
        100,
        empty_callback,
    )

    cv2.createTrackbar(
        "Min side px",
        CONTROLS_WINDOW,
        35,
        500,
        empty_callback,
    )

    # Морфологическое замыкание соединяет белые части,
    # которые были разделены чёрной сеткой.
    cv2.createTrackbar(
        "Close kernel",
        CONTROLS_WINDOW,
        13,
        101,
        empty_callback,
    )

    cv2.createTrackbar(
        "Close iterations",
        CONTROLS_WINDOW,
        2,
        6,
        empty_callback,
    )

    # Opening убирает мелкий белый шум.
    cv2.createTrackbar(
        "Open kernel",
        CONTROLS_WINDOW,
        5,
        51,
        empty_callback,
    )

    cv2.createTrackbar(
        "Open iterations",
        CONTROLS_WINDOW,
        1,
        5,
        empty_callback,
    )


def get_parameters() -> dict[str, float | int]:
    """Считывает текущие значения всех ползунков."""

    saturation_max = cv2.getTrackbarPos(
        "S max",
        CONTROLS_WINDOW,
    )

    value_min = cv2.getTrackbarPos(
        "V min",
        CONTROLS_WINDOW,
    )

    min_area_slider = cv2.getTrackbarPos(
        "Min area x10000",
        CONTROLS_WINDOW,
    )

    max_area_percent = cv2.getTrackbarPos(
        "Max area percent",
        CONTROLS_WINDOW,
    )

    max_aspect_slider = cv2.getTrackbarPos(
        "Max aspect x100",
        CONTROLS_WINDOW,
    )

    min_rectangularity_slider = cv2.getTrackbarPos(
        "Min rectangularity",
        CONTROLS_WINDOW,
    )

    min_side_px = cv2.getTrackbarPos(
        "Min side px",
        CONTROLS_WINDOW,
    )

    close_kernel_slider = cv2.getTrackbarPos(
        "Close kernel",
        CONTROLS_WINDOW,
    )

    close_iterations = cv2.getTrackbarPos(
        "Close iterations",
        CONTROLS_WINDOW,
    )

    open_kernel_slider = cv2.getTrackbarPos(
        "Open kernel",
        CONTROLS_WINDOW,
    )

    open_iterations = cv2.getTrackbarPos(
        "Open iterations",
        CONTROLS_WINDOW,
    )

    return {
        "WHITE_SATURATION_MAX": saturation_max,
        "WHITE_VALUE_MIN": value_min,
        "MIN_WHITE_RECT_AREA_RATIO": (
            min_area_slider / 10000.0
        ),
        "MAX_WHITE_RECT_AREA_RATIO": (
            max_area_percent / 100.0
        ),
        "MAX_WHITE_RECT_ASPECT_RATIO": max(
            1.0,
            max_aspect_slider / 100.0,
        ),
        "MIN_RECTANGULARITY": (
            min_rectangularity_slider / 100.0
        ),
        "MIN_RECT_SIDE_PX": max(
            1,
            min_side_px,
        ),
        "CLOSE_KERNEL": odd_kernel(
            close_kernel_slider
        ),
        "CLOSE_ITERATIONS": close_iterations,
        "OPEN_KERNEL": odd_kernel(
            open_kernel_slider
        ),
        "OPEN_ITERATIONS": open_iterations,
    }


def put_text(
    image: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.6,
) -> None:
    """Рисует читаемый текст с чёрной обводкой."""

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def create_white_mask(
    frame: np.ndarray,
    parameters: dict[str, float | int],
) -> np.ndarray:
    """Создаёт маску белых областей."""

    hsv = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV,
    )

    lower_white = np.array(
        [
            0,
            0,
            int(parameters["WHITE_VALUE_MIN"]),
        ],
        dtype=np.uint8,
    )

    upper_white = np.array(
        [
            179,
            int(parameters["WHITE_SATURATION_MAX"]),
            255,
        ],
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower_white,
        upper_white,
    )

    close_iterations = int(
        parameters["CLOSE_ITERATIONS"]
    )

    if close_iterations > 0:
        close_kernel_size = int(
            parameters["CLOSE_KERNEL"]
        )

        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                close_kernel_size,
                close_kernel_size,
            ),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=close_iterations,
        )

    open_iterations = int(
        parameters["OPEN_ITERATIONS"]
    )

    if open_iterations > 0:
        open_kernel_size = int(
            parameters["OPEN_KERNEL"]
        )

        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                open_kernel_size,
                open_kernel_size,
            ),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=open_iterations,
        )

    return mask


def find_rectangle_candidates(
    frame: np.ndarray,
    mask: np.ndarray,
    parameters: dict[str, float | int],
) -> list[dict]:
    """Ищет контуры, подходящие по площади и форме."""

    frame_height, frame_width = frame.shape[:2]
    frame_area = float(
        frame_height * frame_width
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    candidates: list[dict] = []

    for contour in contours:
        contour_area = float(
            cv2.contourArea(contour)
        )

        if contour_area <= 0:
            continue

        area_ratio = contour_area / frame_area

        if area_ratio < float(
            parameters["MIN_WHITE_RECT_AREA_RATIO"]
        ):
            continue

        if area_ratio > float(
            parameters["MAX_WHITE_RECT_AREA_RATIO"]
        ):
            continue

        rotated_rectangle = cv2.minAreaRect(
            contour
        )

        width, height = rotated_rectangle[1]

        if width <= 0 or height <= 0:
            continue

        if min(width, height) < int(
            parameters["MIN_RECT_SIDE_PX"]
        ):
            continue

        aspect_ratio = (
            max(width, height)
            / min(width, height)
        )

        if aspect_ratio > float(
            parameters[
                "MAX_WHITE_RECT_ASPECT_RATIO"
            ]
        ):
            continue

        rectangle_area = width * height

        if rectangle_area <= 0:
            continue

        rectangularity = (
            contour_area / rectangle_area
        )

        if rectangularity < float(
            parameters["MIN_RECTANGULARITY"]
        ):
            continue

        box = cv2.boxPoints(
            rotated_rectangle
        ).astype(np.int32)

        score = (
            area_ratio
            * rectangularity
            / max(aspect_ratio, 1.0)
        )

        candidates.append(
            {
                "box": box,
                "area_px": contour_area,
                "area_ratio": area_ratio,
                "aspect_ratio": aspect_ratio,
                "rectangularity": rectangularity,
                "score": score,
            }
        )

    candidates.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return candidates


def process_frame(
    frame: np.ndarray,
    parameters: dict[str, float | int],
    paused: bool,
    frame_index: int,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Обрабатывает кадр и рисует подходящие прямоугольники."""

    output = frame.copy()

    mask = create_white_mask(
        frame,
        parameters,
    )

    candidates = find_rectangle_candidates(
        frame,
        mask,
        parameters,
    )

    # Все подходящие кандидаты рисуем тонкой зелёной линией.
    for candidate in candidates:
        cv2.polylines(
            output,
            [candidate["box"]],
            True,
            (0, 255, 0),
            2,
        )

    # Лучшего кандидата выделяем толстой жёлтой рамкой.
    if candidates:
        best = candidates[0]

        cv2.polylines(
            output,
            [best["box"]],
            True,
            (0, 255, 255),
            5,
        )

        center = best["box"].mean(
            axis=0
        ).astype(int)

        cv2.circle(
            output,
            tuple(center),
            7,
            (0, 0, 255),
            -1,
        )

        put_text(
            output,
            "BEST WHITE RECTANGLE",
            (20, 40),
            (0, 255, 255),
        )

        put_text(
            output,
            f"area={best['area_px']:.0f}px "
            f"ratio={best['area_ratio']:.4f}",
            (20, 70),
            (0, 255, 255),
        )

        put_text(
            output,
            f"aspect={best['aspect_ratio']:.2f} "
            f"rect={best['rectangularity']:.2f}",
            (20, 100),
            (0, 255, 255),
        )

    else:
        put_text(
            output,
            "NO WHITE RECTANGLE",
            (20, 40),
            (0, 0, 255),
        )

    put_text(
        output,
        f"candidates={len(candidates)}",
        (20, 130),
        (255, 255, 255),
    )

    put_text(
        output,
        (
            f"Smax={parameters['WHITE_SATURATION_MAX']} "
            f"Vmin={parameters['WHITE_VALUE_MIN']}"
        ),
        (20, 160),
        (255, 255, 255),
    )

    put_text(
        output,
        (
            f"area="
            f"{parameters['MIN_WHITE_RECT_AREA_RATIO']:.4f}-"
            f"{parameters['MAX_WHITE_RECT_AREA_RATIO']:.2f}"
        ),
        (20, 190),
        (255, 255, 255),
    )

    put_text(
        output,
        (
            f"aspect<="
            f"{parameters['MAX_WHITE_RECT_ASPECT_RATIO']:.2f} "
            f"rect>="
            f"{parameters['MIN_RECTANGULARITY']:.2f}"
        ),
        (20, 220),
        (255, 255, 255),
    )

    time_seconds = (
        frame_index / fps
        if fps > 0
        else 0.0
    )

    status = "PAUSED" if paused else "PLAY"

    put_text(
        output,
        (
            f"{status} frame={frame_index} "
            f"time={time_seconds:.2f}s"
        ),
        (20, output.shape[0] - 25),
        (255, 255, 255),
    )

    return output, mask


def print_parameters(
    parameters: dict[str, float | int],
) -> None:
    """Печатает параметры в удобном для копирования виде."""

    print("\n" + "=" * 60)
    print("ТЕКУЩИЕ ПАРАМЕТРЫ")
    print("=" * 60)

    for name, value in parameters.items():
        if isinstance(value, float):
            print(f"{name} = {value:.6f}")
        else:
            print(f"{name} = {value}")

    print("\nJSON:")

    print(
        json.dumps(
            parameters,
            ensure_ascii=False,
            indent=4,
        )
    )

    print("=" * 60 + "\n")


def save_parameters(
    parameters: dict[str, float | int],
    output_path: Path,
) -> None:
    """Сохраняет параметры в JSON."""

    output_path.write_text(
        json.dumps(
            parameters,
            ensure_ascii=False,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        f"Параметры сохранены: {output_path}"
    )


def main() -> int:
    args = parse_args()

    video_path = choose_video(
        args.video
    )

    if video_path is None:
        print("Видео не выбрано.")
        return 1

    capture, fps, frame_count = open_video(
        video_path
    )

    seek_state = {
        "frame": None,
        "updating": False,
    }

    create_windows(
        frame_count,
        seek_state,
    )

    paused = True
    current_frame: np.ndarray | None = None

    ok, current_frame = capture.read()

    if not ok or current_frame is None:
        capture.release()
        cv2.destroyAllWindows()

        raise RuntimeError(
            "Не удалось прочитать первый кадр"
        )

    print(f"Видео: {video_path}")
    print(f"FPS: {fps:.2f}")
    print(f"Кадров: {frame_count}")

    print("\nУправление:")
    print("  Space — пауза / продолжить")
    print("  A/D   — назад / вперёд на 1 секунду")
    print("  J/L   — назад / вперёд на 5 секунд")
    print("  R     — перейти в начало")
    print("  S     — вывести текущие параметры")
    print("  W     — сохранить параметры в JSON")
    print("  Q/Esc — выход")

    parameters_path = (
        video_path.parent
        / "contour_thresholds.json"
    )

    try:
        while True:
            requested_frame = seek_state["frame"]

            if requested_frame is not None:
                frame_number = max(
                    0,
                    int(requested_frame),
                )

                if frame_count > 0:
                    frame_number = min(
                        frame_number,
                        frame_count - 1,
                    )

                capture.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    frame_number,
                )

                ok, frame = capture.read()

                seek_state["frame"] = None

                if ok and frame is not None:
                    current_frame = frame
                    paused = True

            elif not paused:
                ok, frame = capture.read()

                if ok and frame is not None:
                    current_frame = frame
                else:
                    paused = True

            frame_index = max(
                0,
                int(
                    capture.get(
                        cv2.CAP_PROP_POS_FRAMES
                    )
                ) - 1,
            )

            parameters = get_parameters()

            output, mask = process_frame(
                current_frame,
                parameters,
                paused,
                frame_index,
                fps,
            )

            cv2.imshow(
                VIDEO_WINDOW,
                output,
            )

            cv2.imshow(
                MASK_WINDOW,
                mask,
            )

            if frame_count > 0:
                seek_state["updating"] = True

                cv2.setTrackbarPos(
                    FRAME_TRACKBAR,
                    VIDEO_WINDOW,
                    min(
                        frame_index,
                        frame_count - 1,
                    ),
                )

                seek_state["updating"] = False

            delay = (
                30
                if paused
                else max(
                    1,
                    int(1000 / fps),
                )
            )

            key = cv2.waitKey(delay) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key == ord(" "):
                paused = not paused

            elif key in (ord("s"), ord("S")):
                print_parameters(
                    parameters
                )

            elif key in (ord("w"), ord("W")):
                save_parameters(
                    parameters,
                    parameters_path,
                )

            elif key in (ord("r"), ord("R")):
                seek_state["frame"] = 0

            elif key in (
                ord("a"),
                ord("A"),
                ord("d"),
                ord("D"),
                ord("j"),
                ord("J"),
                ord("l"),
                ord("L"),
            ):
                one_second = max(
                    1,
                    int(round(fps)),
                )

                if key in (ord("a"), ord("A")):
                    seek_state["frame"] = (
                        frame_index - one_second
                    )

                elif key in (ord("d"), ord("D")):
                    seek_state["frame"] = (
                        frame_index + one_second
                    )

                elif key in (ord("j"), ord("J")):
                    seek_state["frame"] = (
                        frame_index
                        - 5 * one_second
                    )

                else:
                    seek_state["frame"] = (
                        frame_index
                        + 5 * one_second
                    )

    finally:
        # При закрытии всегда печатаем итоговые параметры.
        try:
            final_parameters = get_parameters()
            print_parameters(
                final_parameters
            )
        except Exception:
            pass

        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())