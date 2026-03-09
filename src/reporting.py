from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import cv2
import matplotlib.pyplot as plt
import numpy as np

from utils import draw_text, make_montage, resize_for_display, sample_indices

def save_image(path: Path, image: np.ndarray):
    cv2.imwrite(str(path), image)


def save_lk_grid(vis_frames, output_dir: Path, title: str = "LK trajectories"):
    idxs = sample_indices(len(vis_frames), 6)
    frames = [resize_for_display(vis_frames[i], width=480) for i in idxs]
    grid = make_montage(frames, rows=2, cols=3)
    grid = draw_text(grid, title)
    save_image(output_dir / "lk_tracks_grid.png", grid)


def save_farneback_grid(flow_frames, output_dir: Path, title: str = "Farneback flow"):
    idxs = sample_indices(len(flow_frames), 6)
    frames = [resize_for_display(flow_frames[i], width=480) for i in idxs]
    grid = make_montage(frames, rows=2, cols=3)
    grid = draw_text(grid, title)
    save_image(output_dir / "farneback_flow_grid.png", grid)


def save_mask_grid(mask_frames, output_dir: Path, title: str = "Motion masks"):
    idxs = sample_indices(len(mask_frames), 6)
    frames = []
    for i in idxs:
        mask_bgr = cv2.cvtColor(mask_frames[i], cv2.COLOR_GRAY2BGR)
        frames.append(resize_for_display(mask_bgr, width=480))
    grid = make_montage(frames, rows=2, cols=3)
    grid = draw_text(grid, title)
    save_image(output_dir / "motion_masks_grid.png", grid)


def save_lk_metrics(metrics, output_dir: Path):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(metrics["num_points"], label="active points")
    plt.plot(metrics["lost_points"], label="lost points")
    plt.plot(metrics["mean_displacement"], label="mean displacement")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title("Lucas-Kanade metrics")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "lk_metrics.png", dpi=180)
    plt.close(fig)


def save_farneback_metrics(metrics, output_dir: Path):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(metrics["mean_magnitude"], label="mean magnitude")
    plt.plot(metrics["motion_area_ratio"], label="motion area ratio")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title("Farneback metrics")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "farneback_metrics.png", dpi=180)
    plt.close(fig)


def save_comparison_video(
    original_frames,
    lk_vis_frames,
    flow_frames,
    mask_frames,
    output_path: Path,
    fps: float,
):
    n = min(len(lk_vis_frames), len(flow_frames), len(mask_frames), len(original_frames) - 1)
    if n <= 0:
        return

    sample = original_frames[1]
    h, w = sample.shape[:2]
    out_w = w * 2
    out_h = h * 2

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    for i in range(n):
        original = original_frames[i + 1]
        lk = cv2.resize(lk_vis_frames[i], (w, h))
        flow = cv2.resize(flow_frames[i], (w, h))
        mask = cv2.cvtColor(mask_frames[i], cv2.COLOR_GRAY2BGR)
        mask = cv2.resize(mask, (w, h))

        original = draw_text(original, "Original", (20, 30))
        lk = draw_text(lk, "Lucas-Kanade", (20, 30))
        flow = draw_text(flow, "Farneback", (20, 30))
        mask = draw_text(mask, "Motion mask", (20, 30))

        top = np.hstack([original, lk])
        bottom = np.hstack([flow, mask])
        canvas = np.vstack([top, bottom])

        writer.write(canvas)

    writer.release()


def save_analysis_md(lk_metrics, fb_metrics, output_dir: Path):
    mean_points = float(np.mean(lk_metrics["num_points"])) if lk_metrics["num_points"] else 0.0
    mean_lost = float(np.mean(lk_metrics["lost_points"])) if lk_metrics["lost_points"] else 0.0
    mean_disp = float(np.mean(lk_metrics["mean_displacement"])) if lk_metrics["mean_displacement"] else 0.0

    mean_mag = float(np.mean(fb_metrics["mean_magnitude"])) if fb_metrics["mean_magnitude"] else 0.0
    mean_area = float(np.mean(fb_metrics["motion_area_ratio"])) if fb_metrics["motion_area_ratio"] else 0.0

    text = f"""# Analysis

## Lucas-Kanade

- Mean number of active points: {mean_points:.2f}
- Mean number of lost points per frame: {mean_lost:.2f}
- Mean displacement: {mean_disp:.2f}

Observations:
- Lucas–Kanade gives interpretable trajectories and works well on textured regions.
- The tracker becomes less stable in weakly textured areas and under motion blur.
- Points may disappear during fast motion, occlusions and abrupt illumination changes.

## Farneback

- Mean flow magnitude: {mean_mag:.2f}
- Mean motion area ratio: {mean_area:.4f}

Observations:
- Farnebäck captures motion densely across the frame.
- It is suitable for obtaining motion masks and highlighting moving regions.
- The method may produce noisy motion estimates near shadows, blur and low-contrast boundaries.

## Comparison

- Lucas–Kanade is better for feature-level trajectory analysis.
- Farnebäck is better for dense motion estimation and segmentation-like motion masks.
- In real applications, the choice depends on whether the task requires sparse stable tracks or dense motion fields.
"""
    (output_dir / "analysis.md").write_text(text, encoding="utf-8")