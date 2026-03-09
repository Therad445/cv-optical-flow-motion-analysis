from pathlib import Path

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_video_frames(video_path: str, max_frames: int | None = None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 25.0

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
        count += 1
        if max_frames is not None and count >= max_frames:
            break

    cap.release()

    if not frames:
        raise ValueError("No frames were read from the video")

    return frames, fps


def to_gray(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def resize_for_display(frame: np.ndarray, width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(frame, new_size)


def make_montage(images, rows: int, cols: int):
    if not images:
        raise ValueError("Empty image list")

    h, w = images[0].shape[:2]
    channels = 1 if images[0].ndim == 2 else images[0].shape[2]

    if channels == 1:
        canvas = np.zeros((rows * h, cols * w), dtype=images[0].dtype)
    else:
        canvas = np.zeros((rows * h, cols * w, channels), dtype=images[0].dtype)

    for idx, img in enumerate(images[: rows * cols]):
        r = idx // cols
        c = idx % cols
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = img

    return canvas


def sample_indices(n: int, k: int = 6):
    if n <= k:
        return list(range(n))
    return np.linspace(0, n - 1, k, dtype=int).tolist()


def draw_text(frame: np.ndarray, text: str, pos=(20, 30), scale: float = 0.8):
    out = frame.copy()
    cv2.putText(
        out,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        text,
        pos,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out