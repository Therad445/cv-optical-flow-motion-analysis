import argparse
from pathlib import Path

from farneback_motion import FarnebackMotion
from lk_tracker import LKTracker
from reporting import (
    save_analysis_md,
    save_comparison_video,
    save_farneback_grid,
    save_farneback_metrics,
    save_lk_grid,
    save_lk_metrics,
    save_mask_grid,
)
from utils import ensure_dir, read_video_frames, to_gray


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input video")
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--max-frames", type=int, default=180, help="Max number of frames")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = ensure_dir(args.output)

    frames, fps = read_video_frames(args.input, max_frames=args.max_frames)
    gray_frames = [to_gray(frame) for frame in frames]

    lk_tracker = LKTracker()
    lk_result = lk_tracker.track(gray_frames)

    farneback = FarnebackMotion()
    fb_result = farneback.process(gray_frames)

    save_lk_grid(lk_result["vis_frames"], output_dir)
    save_farneback_grid(fb_result["flow_frames"], output_dir)
    save_mask_grid(fb_result["mask_frames"], output_dir)

    save_lk_metrics(lk_result["metrics"], output_dir)
    save_farneback_metrics(fb_result["metrics"], output_dir)

    save_comparison_video(
        original_frames=frames,
        lk_vis_frames=lk_result["vis_frames"],
        flow_frames=fb_result["flow_frames"],
        mask_frames=fb_result["mask_frames"],
        output_path=Path(output_dir) / "comparison.mp4",
        fps=fps,
    )

    save_analysis_md(lk_result["metrics"], fb_result["metrics"], Path(output_dir))

    print(f"Done. Results saved to: {output_dir}")


if __name__ == "__main__":
    main()