import argparse
from pathlib import Path

import cv2


def extract_frames(video_path: Path, output_dir: Path, every: int = 1) -> int:
    """Сохраняет каждый N-й кадр видео и возвращает число сохранённых кадров."""
    if every < 1:
        raise ValueError("Параметр --every должен быть не меньше 1")

    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    number_width = max(6, len(str(max(total_frames, 1))))
    frame_index = 0
    saved_count = 0

    try:
        while True:
            success, frame = video.read()
            if not success:
                break

            if frame_index % every == 0:
                frame_number = frame_index + 1
                output_path = output_dir / (
                    f"frame_{frame_number:0{number_width}d}.jpg"
                )

                if not cv2.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Не удалось сохранить кадр: {output_path}")

                saved_count += 1

            frame_index += 1
    finally:
        video.release()

    return saved_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Разбивает видео на отдельные JPG-кадры."
    )
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=Path("flight.mp4"),
        help="путь к видео (по умолчанию: flight.mp4)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("frames"),
        help="папка для кадров (по умолчанию: frames)",
    )
    parser.add_argument(
        "-e",
        "--every",
        type=int,
        default=1,
        help="сохранять каждый N-й кадр (по умолчанию: 1)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.is_file():
        raise SystemExit(f"Видео не найдено: {args.video}")

    try:
        saved_count = extract_frames(args.video, args.output, args.every)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(f"Ошибка: {error}") from error

    print(f"Готово: сохранено кадров — {saved_count}")
    print(f"Папка: {args.output.resolve()}")


if __name__ == "__main__":
    main()
