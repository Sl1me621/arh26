import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


VIDEO_WINDOW = "video"
MASK_WINDOW = "mask"
OVERLAY_WINDOW = "overlay"
CONTROLS_WINDOW = "controls"

DEFAULT_MASKS = {
    "red": {
        "lower": [0, 80, 80],
        "upper": [10, 255, 255],
        "lower2": [170, 80, 80],
        "upper2": [179, 255, 255],
        "use_second_range": True,
    },
    "blue": {
        "lower": [90, 70, 50],
        "upper": [130, 255, 255],
        "lower2": [0, 0, 0],
        "upper2": [0, 0, 0],
        "use_second_range": False,
    },
    "white": {
        "lower": [0, 0, 180],
        "upper": [179, 70, 255],
        "lower2": [0, 0, 0],
        "upper2": [0, 0, 0],
        "use_second_range": False,
    },
    "yellow": {
        "lower": [20, 80, 80],
        "upper": [38, 255, 255],
        "lower2": [0, 0, 0],
        "upper2": [0, 0, 0],
        "use_second_range": False,
    },
    "green": {
        "lower": [40, 60, 50],
        "upper": [85, 255, 255],
        "lower2": [0, 0, 0],
        "upper2": [0, 0, 0],
        "use_second_range": False,
    },
    "orange": {
        "lower": [5, 100, 80],
        "upper": [22, 255, 255],
        "lower2": [0, 0, 0],
        "upper2": [0, 0, 0],
        "use_second_range": False,
    },
}


def nothing(_: int) -> None:
    pass


def clamp_hsv(value: list[int]) -> list[int]:
    return [
        int(np.clip(value[0], 0, 179)),
        int(np.clip(value[1], 0, 255)),
        int(np.clip(value[2], 0, 255)),
    ]


def normalize_mask_config(raw: dict[str, Any]) -> dict[str, Any]:
    lower = clamp_hsv(raw.get("lower", [0, 0, 0]))
    upper = clamp_hsv(raw.get("upper", [179, 255, 255]))
    lower2 = clamp_hsv(raw.get("lower2", [0, 0, 0]))
    upper2 = clamp_hsv(raw.get("upper2", [0, 0, 0]))
    return {
        "lower": lower,
        "upper": upper,
        "lower2": lower2,
        "upper2": upper2,
        "use_second_range": bool(raw.get("use_second_range", False)),
    }


def load_masks(path: Path) -> dict[str, dict[str, Any]]:
    masks = {name: normalize_mask_config(config) for name, config in DEFAULT_MASKS.items()}
    if not path.exists():
        return masks

    with path.open("r", encoding="utf-8") as file:
        loaded = json.load(file)

    for name, config in loaded.items():
        if isinstance(config, dict):
            masks[name] = normalize_mask_config(config)
    return masks


def save_masks(path: Path, masks: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(masks, file, indent=2, ensure_ascii=False)
    print(f"Saved masks: {path}")


def create_trackbars(total_frames: int) -> None:
    cv2.namedWindow(CONTROLS_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROLS_WINDOW, 520, 500)

    cv2.createTrackbar("H min", CONTROLS_WINDOW, 0, 179, nothing)
    cv2.createTrackbar("S min", CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("V min", CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("H max", CONTROLS_WINDOW, 179, 179, nothing)
    cv2.createTrackbar("S max", CONTROLS_WINDOW, 255, 255, nothing)
    cv2.createTrackbar("V max", CONTROLS_WINDOW, 255, 255, nothing)

    cv2.createTrackbar("H2 min", CONTROLS_WINDOW, 0, 179, nothing)
    cv2.createTrackbar("S2 min", CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("V2 min", CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("H2 max", CONTROLS_WINDOW, 0, 179, nothing)
    cv2.createTrackbar("S2 max", CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("V2 max", CONTROLS_WINDOW, 0, 255, nothing)
    cv2.createTrackbar("Use H2", CONTROLS_WINDOW, 0, 1, nothing)

    cv2.createTrackbar("Blur", CONTROLS_WINDOW, 1, 25, nothing)
    cv2.createTrackbar("Open", CONTROLS_WINDOW, 0, 20, nothing)
    cv2.createTrackbar("Close", CONTROLS_WINDOW, 0, 20, nothing)
    cv2.createTrackbar("Min area", CONTROLS_WINDOW, 50, 5000, nothing)
    cv2.createTrackbar("Overlay %", CONTROLS_WINDOW, 45, 100, nothing)
    cv2.createTrackbar("Frame", CONTROLS_WINDOW, 0, max(1, total_frames - 1), nothing)


def set_trackbar_config(config: dict[str, Any]) -> None:
    lower = config["lower"]
    upper = config["upper"]
    lower2 = config["lower2"]
    upper2 = config["upper2"]

    cv2.setTrackbarPos("H min", CONTROLS_WINDOW, lower[0])
    cv2.setTrackbarPos("S min", CONTROLS_WINDOW, lower[1])
    cv2.setTrackbarPos("V min", CONTROLS_WINDOW, lower[2])
    cv2.setTrackbarPos("H max", CONTROLS_WINDOW, upper[0])
    cv2.setTrackbarPos("S max", CONTROLS_WINDOW, upper[1])
    cv2.setTrackbarPos("V max", CONTROLS_WINDOW, upper[2])

    cv2.setTrackbarPos("H2 min", CONTROLS_WINDOW, lower2[0])
    cv2.setTrackbarPos("S2 min", CONTROLS_WINDOW, lower2[1])
    cv2.setTrackbarPos("V2 min", CONTROLS_WINDOW, lower2[2])
    cv2.setTrackbarPos("H2 max", CONTROLS_WINDOW, upper2[0])
    cv2.setTrackbarPos("S2 max", CONTROLS_WINDOW, upper2[1])
    cv2.setTrackbarPos("V2 max", CONTROLS_WINDOW, upper2[2])
    cv2.setTrackbarPos("Use H2", CONTROLS_WINDOW, int(config["use_second_range"]))


def read_trackbar_config() -> dict[str, Any]:
    return {
        "lower": [
            cv2.getTrackbarPos("H min", CONTROLS_WINDOW),
            cv2.getTrackbarPos("S min", CONTROLS_WINDOW),
            cv2.getTrackbarPos("V min", CONTROLS_WINDOW),
        ],
        "upper": [
            cv2.getTrackbarPos("H max", CONTROLS_WINDOW),
            cv2.getTrackbarPos("S max", CONTROLS_WINDOW),
            cv2.getTrackbarPos("V max", CONTROLS_WINDOW),
        ],
        "lower2": [
            cv2.getTrackbarPos("H2 min", CONTROLS_WINDOW),
            cv2.getTrackbarPos("S2 min", CONTROLS_WINDOW),
            cv2.getTrackbarPos("V2 min", CONTROLS_WINDOW),
        ],
        "upper2": [
            cv2.getTrackbarPos("H2 max", CONTROLS_WINDOW),
            cv2.getTrackbarPos("S2 max", CONTROLS_WINDOW),
            cv2.getTrackbarPos("V2 max", CONTROLS_WINDOW),
        ],
        "use_second_range": bool(cv2.getTrackbarPos("Use H2", CONTROLS_WINDOW)),
    }


def odd_kernel_size(value: int) -> int:
    if value <= 1:
        return 1
    return value if value % 2 == 1 else value + 1


def morphology_kernel(value: int) -> np.ndarray | None:
    if value <= 0:
        return None
    size = value if value % 2 == 1 else value + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_mask(frame: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    blur = odd_kernel_size(cv2.getTrackbarPos("Blur", CONTROLS_WINDOW))
    source = cv2.GaussianBlur(frame, (blur, blur), 0) if blur > 1 else frame
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        np.array(config["lower"], dtype=np.uint8),
        np.array(config["upper"], dtype=np.uint8),
    )
    if config["use_second_range"]:
        mask2 = cv2.inRange(
            hsv,
            np.array(config["lower2"], dtype=np.uint8),
            np.array(config["upper2"], dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, mask2)

    open_kernel = morphology_kernel(cv2.getTrackbarPos("Open", CONTROLS_WINDOW))
    if open_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    close_kernel = morphology_kernel(cv2.getTrackbarPos("Close", CONTROLS_WINDOW))
    if close_kernel is not None:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask


def draw_detections(frame: np.ndarray, mask: np.ndarray, color_name: str) -> np.ndarray:
    result = frame.copy()
    min_area = cv2.getTrackbarPos("Min area", CONTROLS_WINDOW)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if radius < 2:
            continue

        center = (int(x), int(y))
        cv2.circle(result, center, int(radius), (0, 255, 0), 2)
        cv2.circle(result, center, 3, (0, 0, 255), -1)
        cv2.putText(
            result,
            f"{color_name} {int(area)}",
            (max(0, center[0] - 35), max(20, center[1] - int(radius) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return result


def make_overlay(frame: np.ndarray, mask: np.ndarray, color_name: str) -> np.ndarray:
    overlay_strength = cv2.getTrackbarPos("Overlay %", CONTROLS_WINDOW) / 100.0
    colored = np.zeros_like(frame)
    colored[mask > 0] = (0, 255, 255)
    overlay = cv2.addWeighted(frame, 1.0, colored, overlay_strength, 0)
    overlay = draw_detections(overlay, mask, color_name)
    return overlay


def draw_status(frame: np.ndarray, color_name: str, frame_index: int, total_frames: int, paused: bool) -> np.ndarray:
    result = frame.copy()
    status = "pause" if paused else "play"
    text = f"{color_name} | {status} | frame {frame_index + 1}/{total_frames}"
    cv2.rectangle(result, (8, 8), (8 + min(720, 12 * len(text)), 38), (0, 0, 0), -1)
    cv2.putText(
        result,
        text,
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return result


def print_help(colors: list[str], masks_path: Path) -> None:
    print()
    print("Controls:")
    print("  Space      pause/play")
    print("  A / D      previous/next frame while paused")
    print("  Z / X      rewind/forward by 1 second")
    print("  [ / ]      previous/next color preset")
    print("  1..9       select color preset by number")
    print("  S          save HSV masks to JSON")
    print("  H          print this help")
    print("  Q or Esc   quit")
    print()
    print("Color presets:")
    for index, color in enumerate(colors, start=1):
        print(f"  {index}: {color}")
    print()
    print(f"Masks file: {masks_path}")
    print()


def open_video(path: Path) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return capture


def clamp_frame(index: int, total_frames: int) -> int:
    return int(np.clip(index, 0, max(0, total_frames - 1)))


def tune_masks(video_path: Path, masks_path: Path) -> None:
    masks = load_masks(masks_path)
    colors = list(masks.keys())
    color_index = 0
    current_color = colors[color_index]

    capture = open_video(video_path)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_index = 0
    requested_frame = 0
    paused = True

    cv2.namedWindow(VIDEO_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(MASK_WINDOW, cv2.WINDOW_NORMAL)
    cv2.namedWindow(OVERLAY_WINDOW, cv2.WINDOW_NORMAL)
    create_trackbars(total_frames)
    set_trackbar_config(masks[current_color])
    print_help(colors, masks_path)

    try:
        while True:
            slider_frame = cv2.getTrackbarPos("Frame", CONTROLS_WINDOW)
            if slider_frame != requested_frame:
                frame_index = clamp_frame(slider_frame, total_frames)
                requested_frame = frame_index
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                paused = True

            ok, frame = capture.read()
            if not ok or frame is None:
                frame_index = 0
                requested_frame = 0
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            frame_index = clamp_frame(frame_index, total_frames)
            requested_frame = frame_index
            cv2.setTrackbarPos("Frame", CONTROLS_WINDOW, frame_index)

            masks[current_color] = normalize_mask_config(read_trackbar_config())
            mask = build_mask(frame, masks[current_color])
            overlay = make_overlay(frame, mask, current_color)
            frame_with_status = draw_status(frame, current_color, frame_index, total_frames, paused)

            cv2.imshow(VIDEO_WINDOW, frame_with_status)
            cv2.imshow(MASK_WINDOW, mask)
            cv2.imshow(OVERLAY_WINDOW, overlay)

            if paused:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                delay = 30
            else:
                delay = max(1, int(1000 / fps))

            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord("s"):
                masks[current_color] = normalize_mask_config(read_trackbar_config())
                save_masks(masks_path, masks)
            elif key == ord("h"):
                print_help(colors, masks_path)
            elif key == ord("a"):
                paused = True
                frame_index = clamp_frame(frame_index - 1, total_frames)
                requested_frame = frame_index
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                cv2.setTrackbarPos("Frame", CONTROLS_WINDOW, frame_index)
            elif key == ord("d"):
                paused = True
                frame_index = clamp_frame(frame_index + 1, total_frames)
                requested_frame = frame_index
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                cv2.setTrackbarPos("Frame", CONTROLS_WINDOW, frame_index)
            elif key == ord("z"):
                paused = True
                frame_index = clamp_frame(frame_index - int(fps), total_frames)
                requested_frame = frame_index
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                cv2.setTrackbarPos("Frame", CONTROLS_WINDOW, frame_index)
            elif key == ord("x"):
                paused = True
                frame_index = clamp_frame(frame_index + int(fps), total_frames)
                requested_frame = frame_index
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                cv2.setTrackbarPos("Frame", CONTROLS_WINDOW, frame_index)
            elif key in (ord("["), ord("]")):
                masks[current_color] = normalize_mask_config(read_trackbar_config())
                step = -1 if key == ord("[") else 1
                color_index = (color_index + step) % len(colors)
                current_color = colors[color_index]
                set_trackbar_config(masks[current_color])
                print(f"Selected color: {current_color}")
            elif ord("1") <= key <= ord("9"):
                selected = key - ord("1")
                if selected < len(colors):
                    masks[current_color] = normalize_mask_config(read_trackbar_config())
                    color_index = selected
                    current_color = colors[color_index]
                    set_trackbar_config(masks[current_color])
                    print(f"Selected color: {current_color}")
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune HSV color masks on an MP4 video with OpenCV trackbars."
    )
    parser.add_argument("video", type=Path, help="Path to input video, for example ./recordings/test.mp4")
    parser.add_argument(
        "--masks",
        type=Path,
        default=Path("color_masks.json"),
        help="JSON file for HSV masks. Default: color_masks.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tune_masks(args.video.resolve(), args.masks.resolve())


if __name__ == "__main__":
    main()
