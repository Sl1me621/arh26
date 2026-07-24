"""Run inference for one image with a trained Keras or TFLite classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict black box digit class for one image.")
    parser.add_argument("--model", type=Path, required=True, help="Path to .h5 or .tflite model.")
    parser.add_argument("--image", type=Path, required=True, help="Path to input image.")
    parser.add_argument("--config", type=Path, default=Path("config.json"), help="Path to config.json.")
    return parser.parse_args()


def load_dependencies():
    try:
        import numpy as np
        import tensorflow as tf
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install requirements first: "
            "python -m pip install -r requirements.txt\n"
            f"Original error: {exc}"
        ) from exc
    return np, tf, Image


def resolve_path(base_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: Path) -> dict:
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def preprocess_image(image_path: Path, image_size: tuple[int, int], np, image_module):
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    with image_module.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(image_size)
        array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_keras(model_path: Path, image_batch, tf):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model.predict(image_batch, verbose=0)[0]


def predict_tflite(model_path: Path, image_batch, tf, np):
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_index = input_details[0]["index"]
    input_dtype = input_details[0]["dtype"]
    model_input = image_batch.astype(input_dtype)
    if input_dtype == np.uint8:
        scale, zero_point = input_details[0]["quantization"]
        if scale:
            model_input = image_batch / scale + zero_point
            model_input = np.clip(model_input, 0, 255).astype(np.uint8)

    interpreter.set_tensor(input_index, model_input)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])[0]


def main() -> int:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent
    model_path = resolve_path(base_dir, args.model)
    image_path = resolve_path(base_dir, args.image)
    config_path = resolve_path(base_dir, args.config)

    np, tf, image_module = load_dependencies()
    config = load_config(config_path)
    class_names = config.get("class_names")
    if not class_names:
        raise ValueError("config.json must contain class_names.")
    image_size = tuple(config.get("model_input_size", [224, 224]))

    image_batch = preprocess_image(image_path, image_size, np, image_module)
    if model_path.suffix.lower() == ".tflite":
        predictions = predict_tflite(model_path, image_batch, tf, np)
    else:
        predictions = predict_keras(model_path, image_batch, tf)

    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 1 or len(predictions) != len(class_names):
        raise ValueError(
            f"Model output shape does not match class_names: "
            f"got {predictions.shape}, classes={len(class_names)}"
        )

    top_indices = predictions.argsort()[-3:][::-1]
    best_index = int(top_indices[0])
    print(f"predicted class: {class_names[best_index]}")
    print(f"confidence: {predictions[best_index]:.6f}")
    print("top-3 predictions:")
    for index in top_indices:
        print(f"  {class_names[int(index)]}: {predictions[int(index)]:.6f}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
