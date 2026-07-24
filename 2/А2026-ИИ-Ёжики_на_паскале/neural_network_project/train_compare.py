
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASS_COUNT = 10
DEFAULT_CLASS_NAMES = [
    "digit_1",
    "digit_2",
    "digit_3",
    "digit_4",
    "digit_5",
    "digit_6",
    "digit_7",
    "digit_8",
    "digit_9",
    "no_object",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune MobileNetV2, train/fine-tune MobileNetV3Small, "
            "compare them, and export the selected final model."
        )
    )
    parser.add_argument("--epochs", type=int, default=20, help="Compatibility alias for --head-epochs.")
    parser.add_argument("--head-epochs", type=int, default=None, help="Epochs for frozen-backbone training.")
    parser.add_argument("--finetune-epochs", type=int, default=12, help="Epochs for fine-tuning.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for head training.")
    parser.add_argument("--finetune-learning-rate", type=float, default=1e-5, help="Learning rate for fine-tuning.")
    parser.add_argument("--finetune-layers", type=int, default=40, help="Backbone layers to unfreeze from the end.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--benchmark-samples", type=int, default=200, help="Samples for inference timing.")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse existing final checkpoints when present instead of training them again.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../dataset_split"),
        help="Dataset split directory with train/val/test folders.",
    )
    parser.add_argument(
        "--original-dataset-dir",
        type=Path,
        default=Path("../dataset"),
        help="Original dataset directory.",
    )
    return parser.parse_args()


def import_training_dependencies() -> dict[str, Any]:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import tensorflow as tf
        from PIL import Image
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            precision_recall_fscore_support,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install requirements first: "
            "python -m pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc

    return {
        "plt": plt,
        "np": np,
        "pd": pd,
        "tf": tf,
        "Image": Image,
        "classification_report": classification_report,
        "confusion_matrix": confusion_matrix,
        "precision_recall_fscore_support": precision_recall_fscore_support,
    }


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    dirs = {
        "models": base_dir / "models",
        "metrics": base_dir / "metrics",
        "plots": base_dir / "plots",
        "tf_saved_model": base_dir / "tf_saved_model",
        "ir_saved_model": base_dir / "ir_saved_model",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def resolve_project_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def discover_classes(data_dir: Path) -> list[str]:
    split_classes: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test"):
        split_dir = data_dir / split_name
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")
        split_classes[split_name] = sorted(path.name for path in split_dir.iterdir() if path.is_dir())

    reference = split_classes["train"]
    for split_name, class_names in split_classes.items():
        if class_names != reference:
            raise ValueError(
                f"Class folders mismatch in {split_name}. Expected {reference}, got {class_names}"
            )

    if len(reference) != EXPECTED_CLASS_COUNT:
        raise ValueError(f"Expected {EXPECTED_CLASS_COUNT} classes, got {len(reference)}: {reference}")
    return reference


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in class_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate_images(data_dir: Path, class_names: list[str], min_resolution: tuple[int, int], image_module) -> None:
    errors: list[str] = []
    min_width, min_height = min_resolution
    total = 0

    for split_name in ("train", "val", "test"):
        for class_name in class_names:
            class_dir = data_dir / split_name / class_name
            files = list_images(class_dir)
            if not files:
                errors.append(f"No images found in {class_dir}")
                continue

            for path in files:
                total += 1
                try:
                    with image_module.open(path) as image:
                        image.verify()
                    with image_module.open(path) as image:
                        width, height = image.size
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"Unreadable image: {path} ({exc})")
                    continue

                if width < min_width or height < min_height:
                    errors.append(f"Image below {min_width}x{min_height}: {path} ({width}x{height})")

    if errors:
        preview = "\n".join(errors[:30])
        suffix = "" if len(errors) <= 30 else f"\n... and {len(errors) - 30} more errors"
        raise ValueError(f"Dataset validation failed:\n{preview}{suffix}")

    print(f"Dataset validation passed: {total} readable images, minimum resolution OK.")


def make_datasets(tf, data_dir: Path, image_size: tuple[int, int], batch_size: int, seed: int):
    train_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir / "train",
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir / "val",
        labels="inferred",
        label_mode="categorical",
        class_names=train_raw.class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )
    test_raw = tf.keras.utils.image_dataset_from_directory(
        data_dir / "test",
        labels="inferred",
        label_mode="categorical",
        class_names=train_raw.class_names,
        color_mode="rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.04),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.12),
        ],
        name="safe_augmentation",
    )

    def augment(images, labels):
        images = augmentation(images, training=True)
        images = tf.image.random_brightness(images, max_delta=18.0)
        images = tf.clip_by_value(images, 0.0, 255.0)
        return images, labels

    autotune = tf.data.AUTOTUNE
    train_ds = train_raw.map(augment, num_parallel_calls=autotune).prefetch(autotune)
    val_ds = val_raw.prefetch(autotune)
    test_ds = test_raw.prefetch(autotune)
    return train_ds, val_ds, test_ds


def compile_model(tf, model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(tf, checkpoint_path: Path, patience: int = 5):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(2, patience // 2),
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def merge_histories(histories: list[Any]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        if history is None:
            continue
        for key, values in history.history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def save_training_curves(history_data: dict[str, list[float]], plots_dir: Path, model_key: str, plt) -> None:
    for metric_name, filename, ylabel in (
        ("accuracy", f"{model_key}_accuracy.png", "Accuracy"),
        ("loss", f"{model_key}_loss.png", "Loss"),
    ):
        plt.figure(figsize=(8, 5))
        plt.plot(history_data.get(metric_name, []), label=f"train {metric_name}")
        plt.plot(history_data.get(f"val_{metric_name}", []), label=f"val {metric_name}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=160)
        plt.close()


def find_backbone_model(tf, model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and len(getattr(layer, "layers", [])) > 20:
            return layer
    raise ValueError("Backbone model layer was not found. Cannot fine-tune safely.")


def freeze_backbone(tf, model) -> None:
    backbone = find_backbone_model(tf, model)
    backbone.trainable = False
    for layer in model.layers:
        if layer is not backbone:
            layer.trainable = True


def unfreeze_last_backbone_layers(tf, model, last_layers: int) -> None:
    backbone = find_backbone_model(tf, model)
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False
    for layer in backbone.layers[-last_layers:]:
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True
    for layer in model.layers:
        if layer is not backbone:
            layer.trainable = True


def train_head(
    *,
    tf,
    model,
    train_ds,
    val_ds,
    checkpoint_path: Path,
    epochs: int,
    learning_rate: float,
):
    freeze_backbone(tf, model)
    compile_model(tf, model, learning_rate)
    print(f"\nTraining frozen head -> {checkpoint_path.name}")
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=make_callbacks(tf, checkpoint_path),
        verbose=1,
    )


def train_finetune(
    *,
    tf,
    model,
    train_ds,
    val_ds,
    checkpoint_path: Path,
    epochs: int,
    learning_rate: float,
    finetune_layers: int,
):
    unfreeze_last_backbone_layers(tf, model, finetune_layers)
    compile_model(tf, model, learning_rate)
    print(f"\nFine-tuning last {finetune_layers} backbone layers -> {checkpoint_path.name}")
    start_checkpoint = checkpoint_path.with_name(f"{checkpoint_path.stem}_start.h5")
    try:
        model.save(str(start_checkpoint), include_optimizer=False)
    except TypeError:
        model.save(str(start_checkpoint))
    initial_val_loss = float(model.evaluate(val_ds, verbose=0)[0])
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=make_callbacks(tf, checkpoint_path),
        verbose=1,
    )
    if checkpoint_path.exists():
        best_model = tf.keras.models.load_model(checkpoint_path, compile=False)
        compile_model(tf, best_model, learning_rate)
        best_val_loss = float(best_model.evaluate(val_ds, verbose=0)[0])
        if best_val_loss > initial_val_loss:
            shutil.copy2(start_checkpoint, checkpoint_path)
            print(
                f"Fine-tuning did not improve val_loss "
                f"({best_val_loss:.6f} > {initial_val_loss:.6f}); restored starting checkpoint."
            )
    else:
        shutil.copy2(start_checkpoint, checkpoint_path)

    if start_checkpoint.exists():
        start_checkpoint.unlink()
    return history


def load_compiled_model(tf, path: Path, learning_rate: float):
    model = tf.keras.models.load_model(path, compile=False)
    compile_model(tf, model, learning_rate)
    return model


def train_mobilenetv2_finetuned(
    *,
    deps: dict[str, Any],
    dirs: dict[str, Path],
    train_ds,
    val_ds,
    args: argparse.Namespace,
    num_classes: int,
):
    from model import build_mobilenetv2

    tf = deps["tf"]
    base_checkpoint = dirs["models"] / "mobilenetv2_best.h5"
    finetuned_checkpoint = dirs["models"] / "mobilenetv2_finetuned_best.h5"
    histories = []
    start = time.perf_counter()

    if args.resume_existing and finetuned_checkpoint.exists():
        print(f"\nUsing existing fine-tuned MobileNetV2 checkpoint: {finetuned_checkpoint}")
        model = load_compiled_model(tf, finetuned_checkpoint, args.finetune_learning_rate)
        return model, 0.0, {}

    if base_checkpoint.exists():
        print(f"\nLoading existing MobileNetV2 checkpoint: {base_checkpoint}")
        model = load_compiled_model(tf, base_checkpoint, args.learning_rate)
    else:
        print("\nMobileNetV2 base checkpoint not found. Training MobileNetV2 from ImageNet weights first.")
        model = build_mobilenetv2(input_shape=(224, 224, 3), num_classes=num_classes, base_trainable=False)
        histories.append(
            train_head(
                tf=tf,
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                checkpoint_path=base_checkpoint,
                epochs=args.head_epochs,
                learning_rate=args.learning_rate,
            )
        )
        model = load_compiled_model(tf, base_checkpoint, args.learning_rate)

    histories.append(
        train_finetune(
            tf=tf,
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_path=finetuned_checkpoint,
            epochs=args.finetune_epochs,
            learning_rate=args.finetune_learning_rate,
            finetune_layers=args.finetune_layers,
        )
    )
    training_time = time.perf_counter() - start
    model = load_compiled_model(tf, finetuned_checkpoint, args.finetune_learning_rate)
    return model, training_time, merge_histories(histories)


def train_mobilenetv3small(
    *,
    deps: dict[str, Any],
    dirs: dict[str, Path],
    train_ds,
    val_ds,
    args: argparse.Namespace,
    num_classes: int,
):
    from model import build_mobilenetv3small

    tf = deps["tf"]
    base_checkpoint = dirs["models"] / "mobilenetv3small_best.h5"
    finetuned_checkpoint = dirs["models"] / "mobilenetv3small_finetuned_best.h5"
    histories = []
    start = time.perf_counter()

    if args.resume_existing and finetuned_checkpoint.exists():
        print(f"\nUsing existing fine-tuned MobileNetV3Small checkpoint: {finetuned_checkpoint}")
        model = load_compiled_model(tf, finetuned_checkpoint, args.finetune_learning_rate)
        return model, 0.0, {}

    if args.resume_existing and base_checkpoint.exists():
        print(f"\nLoading existing MobileNetV3Small head checkpoint: {base_checkpoint}")
        model = load_compiled_model(tf, base_checkpoint, args.learning_rate)
    else:
        model = build_mobilenetv3small(input_shape=(224, 224, 3), num_classes=num_classes, base_trainable=False)
        histories.append(
            train_head(
                tf=tf,
                model=model,
                train_ds=train_ds,
                val_ds=val_ds,
                checkpoint_path=base_checkpoint,
                epochs=args.head_epochs,
                learning_rate=args.learning_rate,
            )
        )
        model = load_compiled_model(tf, base_checkpoint, args.learning_rate)

    histories.append(
        train_finetune(
            tf=tf,
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            checkpoint_path=finetuned_checkpoint,
            epochs=args.finetune_epochs,
            learning_rate=args.finetune_learning_rate,
            finetune_layers=args.finetune_layers,
        )
    )
    training_time = time.perf_counter() - start
    model = load_compiled_model(tf, finetuned_checkpoint, args.finetune_learning_rate)
    return model, training_time, merge_histories(histories)


def collect_predictions(model, dataset, np):
    y_true: list[int] = []
    y_pred: list[int] = []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1).tolist())
        y_true.extend(np.argmax(labels.numpy(), axis=1).tolist())
    return np.array(y_true), np.array(y_pred)


def benchmark_inference(model, dataset, np, sample_count: int) -> tuple[float, float]:
    batches = []
    collected = 0
    for images, _labels in dataset:
        batches.append(images)
        collected += int(images.shape[0])
        if collected >= sample_count:
            break

    if not batches:
        raise ValueError("No test images available for inference benchmark.")

    images = np.concatenate([batch.numpy() for batch in batches], axis=0)[:sample_count]
    model.predict(images[: min(8, len(images))], verbose=0)
    start = time.perf_counter()
    model.predict(images, verbose=0)
    elapsed = time.perf_counter() - start
    ms_per_image = (elapsed / len(images)) * 1000.0
    fps = 1000.0 / ms_per_image if ms_per_image > 0 else 0.0
    return ms_per_image, fps


def save_confusion_matrix(cm, class_names: list[str], metrics_dir: Path, plots_dir: Path, model_key: str, pd, plt) -> None:
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    df.to_csv(metrics_dir / f"{model_key}_confusion_matrix.csv", encoding="utf-8")

    plt.figure(figsize=(9, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_key} confusion matrix")
    plt.colorbar()
    tick_marks = range(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )
    plt.ylabel("True class")
    plt.xlabel("Predicted class")
    plt.tight_layout()
    plt.savefig(plots_dir / f"{model_key}_confusion_matrix.png", dpi=160)
    plt.close()


def save_tf_saved_model(model, export_dir: Path) -> None:
    if export_dir.exists():
        shutil.rmtree(export_dir)
    if hasattr(model, "export"):
        model.export(str(export_dir))
    else:
        model.save(str(export_dir), save_format="tf")


def convert_tflite(tf, saved_model_dir: Path, output_path: Path, quantized: bool) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    if quantized:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    output_path.write_bytes(converter.convert())


def evaluate_and_save(
    *,
    model,
    model_key: str,
    checkpoint_path: Path,
    training_time: float,
    history_data: dict[str, list[float]],
    deps: dict[str, Any],
    dirs: dict[str, Path],
    test_ds,
    class_names: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    np = deps["np"]
    classification_report = deps["classification_report"]
    confusion_matrix = deps["confusion_matrix"]
    precision_recall_fscore_support = deps["precision_recall_fscore_support"]

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    y_true, y_pred = collect_predictions(model, test_ds, np)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (dirs["metrics"] / f"{model_key}_classification_report.txt").write_text(report_text, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    save_confusion_matrix(cm, class_names, dirs["metrics"], dirs["plots"], model_key, deps["pd"], deps["plt"])
    if history_data:
        save_training_curves(history_data, dirs["plots"], model_key, deps["plt"])

    inference_ms, fps = benchmark_inference(model, test_ds, np, args.benchmark_samples)
    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024) if checkpoint_path.exists() else 0.0
    params = int(model.count_params())

    metrics = {
        "model": model_key,
        "checkpoint_path": project_relative(checkpoint_path),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "training_time_seconds": float(training_time),
        "inference_ms_per_image": float(inference_ms),
        "fps": float(fps),
        "model_size_mb": float(model_size_mb),
        "params": params,
    }
    write_json(dirs["metrics"] / f"{model_key}_metrics.json", metrics)
    return metrics


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(project_dir()).as_posix()
    except ValueError:
        return path.as_posix()


def save_comparison(metrics: list[dict[str, Any]], metrics_dir: Path) -> None:
    fieldnames = [
        "model",
        "test_accuracy",
        "test_loss",
        "f1_macro",
        "f1_weighted",
        "inference_ms_per_image",
        "fps",
        "model_size_mb",
        "params",
        "training_time_seconds",
    ]
    with (metrics_dir / "comparison_table.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow({field: item[field] for field in fieldnames})


def save_comparison_plots(metrics: list[dict[str, Any]], plots_dir: Path, plt) -> None:
    names = [item["model"] for item in metrics]
    plot_specs = [
        ("comparison_accuracy.png", "test_accuracy", "Test accuracy"),
        ("comparison_f1_macro.png", "f1_macro", "F1 macro"),
        ("comparison_inference_time.png", "inference_ms_per_image", "Inference ms/image"),
        ("comparison_model_size.png", "model_size_mb", "Model size MB"),
    ]

    for filename, key, ylabel in plot_specs:
        values = [item[key] for item in metrics]
        plt.figure(figsize=(7, 5))
        plt.bar(names, values, color=["#2563eb", "#16a34a"])
        plt.ylabel(ylabel)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / filename, dpi=160)
        plt.close()


def print_results_table(metrics: list[dict[str, Any]]) -> None:
    headers = [
        "Model",
        "Test Accuracy",
        "F1 Macro",
        "F1 Weighted",
        "Inference ms/image",
        "FPS",
        "Model Size MB",
        "Params",
    ]
    rows = [
        [
            item["model"],
            f"{item['test_accuracy']:.4f}",
            f"{item['f1_macro']:.4f}",
            f"{item['f1_weighted']:.4f}",
            f"{item['inference_ms_per_image']:.2f}",
            f"{item['fps']:.2f}",
            f"{item['model_size_mb']:.2f}",
            str(item["params"]),
        ]
        for item in metrics
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    print()
    print(" | ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))


def choose_final_model(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(
        metrics,
        key=lambda item: (
            item["f1_macro"],
            -item["inference_ms_per_image"],
            -item["model_size_mb"],
        ),
        reverse=True,
    )
    first, second = ordered[0], ordered[1]
    if abs(first["f1_macro"] - second["f1_macro"]) < 0.02:
        faster = min([first, second], key=lambda item: item["inference_ms_per_image"])
        speed_gap = abs(first["inference_ms_per_image"] - second["inference_ms_per_image"])
        similar_speed = speed_gap <= max(first["inference_ms_per_image"], second["inference_ms_per_image"]) * 0.10
        if similar_speed:
            return min([first, second], key=lambda item: item["model_size_mb"])
        return faster
    return first


def make_conclusion(metrics: list[dict[str, Any]], selected: dict[str, Any]) -> str:
    by_accuracy = max(metrics, key=lambda item: item["test_accuracy"])
    by_f1 = max(metrics, key=lambda item: item["f1_macro"])
    by_speed = min(metrics, key=lambda item: item["inference_ms_per_image"])
    by_size = min(metrics, key=lambda item: item["model_size_mb"])

    lines = [
        "",
        "Automatic conclusion:",
        f"- More accurate by test accuracy: {by_accuracy['model']} ({by_accuracy['test_accuracy']:.4f}).",
        f"- Better by macro F1: {by_f1['model']} ({by_f1['f1_macro']:.4f}).",
        f"- Faster model: {by_speed['model']} ({by_speed['inference_ms_per_image']:.2f} ms/image, {by_speed['fps']:.2f} FPS).",
        f"- Lighter model: {by_size['model']} ({by_size['model_size_mb']:.2f} MB).",
        f"- Recommended for Raspberry Pi 4 4GB: {selected['model']}.",
        "- Final model saved as neural_network_project/model.h5.",
    ]
    conclusion = "\n".join(lines)
    print(conclusion)
    return conclusion


def export_final_model(tf, model, dirs: dict[str, Path], base_dir: Path, selected: dict[str, Any]) -> Path:
    source = Path(selected["checkpoint_path"])
    if not source.is_absolute():
        source = base_dir / source
    final_h5 = base_dir / "model.h5"
    shutil.copy2(source, final_h5)

    final_saved_model_dir = dirs["tf_saved_model"] / "final_model"
    save_tf_saved_model(model, final_saved_model_dir)
    convert_tflite(tf, final_saved_model_dir, dirs["models"] / "final_model_float32.tflite", quantized=False)
    convert_tflite(tf, final_saved_model_dir, dirs["models"] / "final_model_quantized.tflite", quantized=True)
    return final_h5


def update_config(
    config_path: Path,
    *,
    args: argparse.Namespace,
    class_names: list[str],
    metrics: list[dict[str, Any]],
    selected: dict[str, Any],
    final_model_path: Path,
    conclusion: str,
) -> None:
    config = {
        "project": "underwater_black_box_digit_classifier",
        "task": "10_class_image_classification",
        "selected_model": selected["model"],
        "class_names": class_names,
        "model_input_size": [224, 224],
        "batch_size": args.batch_size,
        "epochs": {
            "head_epochs": args.head_epochs,
            "finetune_epochs": args.finetune_epochs,
        },
        "optimizer": "Adam",
        "learning_rate": {
            "head": args.learning_rate,
            "fine_tuning": args.finetune_learning_rate,
        },
        "target_platform": "Raspberry Pi 4 4GB",
        "test_accuracy": selected["test_accuracy"],
        "f1_macro": selected["f1_macro"],
        "f1_weighted": selected["f1_weighted"],
        "inference_ms_per_image": selected["inference_ms_per_image"],
        "fps": selected["fps"],
        "model_size_mb": selected["model_size_mb"],
        "final_model_path": final_model_path.name,
        "dataset_split_dir": "../dataset_split",
        "original_dataset_dir": "../dataset",
        "image_min_resolution": [640, 480],
        "metrics_summary": {item["model"]: item for item in metrics},
        "conclusion": conclusion.strip(),
    }
    write_json(config_path, config)


def main() -> int:
    args = parse_args()
    if args.head_epochs is None:
        args.head_epochs = args.epochs

    base_dir = project_dir()
    data_dir = resolve_project_path(base_dir, args.data_dir)
    original_dataset_dir = resolve_project_path(base_dir, args.original_dataset_dir)
    dirs = ensure_dirs(base_dir)

    if not original_dataset_dir.is_dir():
        raise FileNotFoundError(f"Original dataset directory not found: {original_dataset_dir}")

    deps = import_training_dependencies()
    tf = deps["tf"]
    tf.keras.utils.set_random_seed(args.seed)

    class_names = discover_classes(data_dir)
    if class_names != DEFAULT_CLASS_NAMES:
        print(f"Warning: class order differs from default: {class_names}")

    validate_images(data_dir, class_names, (640, 480), deps["Image"])
    train_ds, val_ds, test_ds = make_datasets(
        tf,
        data_dir,
        image_size=(224, 224),
        batch_size=args.batch_size,
        seed=args.seed,
    )

    v2_model, v2_time, v2_history = train_mobilenetv2_finetuned(
        deps=deps,
        dirs=dirs,
        train_ds=train_ds,
        val_ds=val_ds,
        args=args,
        num_classes=len(class_names),
    )
    v2_metrics = evaluate_and_save(
        model=v2_model,
        model_key="mobilenetv2_finetuned",
        checkpoint_path=dirs["models"] / "mobilenetv2_finetuned_best.h5",
        training_time=v2_time,
        history_data=v2_history,
        deps=deps,
        dirs=dirs,
        test_ds=test_ds,
        class_names=class_names,
        args=args,
    )

    v3_model, v3_time, v3_history = train_mobilenetv3small(
        deps=deps,
        dirs=dirs,
        train_ds=train_ds,
        val_ds=val_ds,
        args=args,
        num_classes=len(class_names),
    )
    v3_metrics = evaluate_and_save(
        model=v3_model,
        model_key="mobilenetv3small",
        checkpoint_path=dirs["models"] / "mobilenetv3small_finetuned_best.h5",
        training_time=v3_time,
        history_data=v3_history,
        deps=deps,
        dirs=dirs,
        test_ds=test_ds,
        class_names=class_names,
        args=args,
    )

    metrics = [v2_metrics, v3_metrics]
    save_comparison(metrics, dirs["metrics"])
    save_comparison_plots(metrics, dirs["plots"], deps["plt"])
    print_results_table(metrics)

    selected = choose_final_model(metrics)
    selected_model = v2_model if selected["model"] == "mobilenetv2_finetuned" else v3_model
    final_model_path = export_final_model(tf, selected_model, dirs, base_dir, selected)
    conclusion = make_conclusion(metrics, selected)
    update_config(
        base_dir / "config.json",
        args=args,
        class_names=class_names,
        metrics=metrics,
        selected=selected,
        final_model_path=final_model_path,
        conclusion=conclusion,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
