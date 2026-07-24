"""Create dataset.zip with the competition submission dataset layout."""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ../dataset.zip with dataset/black_box and dataset/no_object folders."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("../dataset"), help="Source dataset directory.")
    parser.add_argument("--output", type=Path, default=Path("../dataset.zip"), help="Output zip path.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality for converted images.")
    return parser.parse_args()


def load_pillow():
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required to create JPEG files. Install requirements first: "
            "python -m pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc
    return Image


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def iter_images(path: Path) -> list[Path]:
    return sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_to_jpeg_bytes(path: Path, image_module, quality: int) -> bytes:
    with image_module.open(path) as image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        return buffer.getvalue()


def prepare_submission_dataset(dataset_dir: Path, output_zip: Path, jpeg_quality: int = 95) -> dict[str, int]:
    image_module = load_pillow()
    black_box_dir = dataset_dir / "black_box"
    no_object_dir = dataset_dir / "no_object"

    if not black_box_dir.is_dir():
        raise FileNotFoundError(f"Missing black_box folder: {black_box_dir}")
    if not no_object_dir.is_dir():
        raise FileNotFoundError(f"Missing no_object folder: {no_object_dir}")

    digit_dirs = sorted(path for path in black_box_dir.iterdir() if path.is_dir())
    if not digit_dirs:
        raise FileNotFoundError(f"No digit folders found in: {black_box_dir}")

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    counts = {"black_box": 0, "no_object": 0}

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for digit_dir in digit_dirs:
            images = iter_images(digit_dir)
            for index, source_path in enumerate(images, start=1):
                counts["black_box"] += 1
                archive_name = f"dataset/black_box/{digit_dir.name}_{index:06d}.jpg"
                archive.writestr(archive_name, image_to_jpeg_bytes(source_path, image_module, jpeg_quality))

        for index, source_path in enumerate(iter_images(no_object_dir), start=1):
            counts["no_object"] += 1
            archive_name = f"dataset/no_object/no_object_{index:06d}.jpg"
            archive.writestr(archive_name, image_to_jpeg_bytes(source_path, image_module, jpeg_quality))

    return counts


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    dataset_dir = resolve_path(base_dir, args.dataset_dir)
    output_zip = resolve_path(base_dir, args.output)
    counts = prepare_submission_dataset(dataset_dir, output_zip, args.jpeg_quality)
    print(f"Created: {output_zip}")
    print(f"dataset/black_box: {counts['black_box']} jpg files")
    print(f"dataset/no_object: {counts['no_object']} jpg files")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
