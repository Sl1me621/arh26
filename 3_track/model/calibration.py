from pathlib import Path

import cv2


SOURCE_DIR = Path("dop_dataset")
OUTPUT_DIR = Path("calibration_dataset")

WIDTH = 640
HEIGHT = 640

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in SOURCE_DIR.rglob("*")
        if path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(
            f"Изображения не найдены в {SOURCE_DIR.resolve()}"
        )

    saved_count = 0

    for index, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))

        if image is None:
            print("Не удалось прочитать:", image_path)
            continue

        resized = cv2.resize(
            image,
            (WIDTH, HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )

        output_path = OUTPUT_DIR / f"calibration_{index:05d}.jpg"

        success = cv2.imwrite(
            str(output_path),
            resized,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if not success:
            print("Не удалось сохранить:", output_path)
            continue

        saved_count += 1

    print("Сохранено изображений:", saved_count)
    print("Папка:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()