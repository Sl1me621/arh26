"""Split an image dataset into train/validation/test folders.

Default layout:
    dataset/
        black_box/
            digit_1/
            digit_2/
            ...
            digit_9/
        no_object/
            ...

Output layout:
    dataset_split/
        train/digit_1/...
        val/digit_1/...
        test/digit_1/...
        ...
        train/no_object/...
        val/no_object/...
        test/no_object/...
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split dataset by classes into 70% train, 15% val, 15% test."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset"),
        help="Path to source dataset folder. Default: dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_split"),
        help="Path to output split dataset folder. Default: dataset_split",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output folder before creating a new split.",
    )
    return parser.parse_args()


def find_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_class_dirs(input_dir: Path) -> list[Path]:
    class_dirs: list[Path] = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_dir():
            continue

        if path.name == "black_box":
            nested_classes = sorted(child for child in path.iterdir() if child.is_dir())
            class_dirs.extend(nested_classes)
            continue

        class_dirs.append(path)

    return class_dirs


def split_files(
    files: list[Path],
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[Path], list[Path], list[Path]]:
    train_count = round(len(files) * train_ratio)
    val_count = round(len(files) * val_ratio)
    train_files = files[:train_count]
    val_files = files[train_count : train_count + val_count]
    test_files = files[train_count + val_count :]
    return train_files, val_files, test_files


def copy_or_move(files: list[Path], source_class_dir: Path, target_class_dir: Path, move: bool) -> None:
    operation = shutil.move if move else shutil.copy2

    for source_path in files:
        relative_path = source_path.relative_to(source_class_dir)
        target_path = target_class_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        operation(source_path, target_path)


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.6f}")
    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError("Ratios must be non-negative")


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input dataset folder not found: {input_dir}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_dir}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(output_dir)

    class_dirs = find_class_dirs(input_dir)
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found in: {input_dir}")

    random.seed(args.seed)
    totals = {"train": 0, "val": 0, "test": 0}

    for class_dir in class_dirs:
        files = find_images(class_dir)
        if not files:
            print(f"Skipping empty class folder: {class_dir.name}")
            continue

        random.shuffle(files)
        train_files, val_files, test_files = split_files(
            files,
            args.train_ratio,
            args.val_ratio,
        )

        split_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, split_files_for_class in split_map.items():
            target_class_dir = output_dir / split_name / class_dir.name
            copy_or_move(split_files_for_class, class_dir, target_class_dir, args.move)
            totals[split_name] += len(split_files_for_class)

        print(
            f"{class_dir.name}: "
            f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}"
        )

    print(
        "Done: "
        f"train={totals['train']}, val={totals['val']}, test={totals['test']}. "
        f"Output: {output_dir}"
    )


if __name__ == "__main__":
    main()
