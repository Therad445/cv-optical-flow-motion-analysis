import cv2
import numpy as np


class FarnebackMotion:
    def __init__(
        self,
        pyr_scale: float = 0.5,
        levels: int = 3,
        winsize: int = 15,
        iterations: int = 3,
        poly_n: int = 5,
        poly_sigma: float = 1.2,
        threshold: float = 1.5,
    ):
        self.params = dict(
            pyr_scale=pyr_scale,
            levels=levels,
            winsize=winsize,
            iterations=iterations,
            poly_n=poly_n,
            poly_sigma=poly_sigma,
            flags=0,
        )
        self.threshold = threshold

    @staticmethod
    def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def process(self, gray_frames):
        prev_gray = gray_frames[0]
        flow_frames = []
        mask_frames = []

        metrics = {
            "mean_magnitude": [],
            "motion_area_ratio": [],
        }

        for i in range(1, len(gray_frames)):
            gray = gray_frames[i]

            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **self.params)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            motion_mask = (mag > self.threshold).astype(np.uint8) * 255
            motion_mask = cv2.medianBlur(motion_mask, 5)

            kernel = np.ones((3, 3), np.uint8)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)

            flow_vis = self.flow_to_hsv(flow)

            flow_frames.append(flow_vis)
            mask_frames.append(motion_mask)

            metrics["mean_magnitude"].append(float(np.mean(mag)))
            metrics["motion_area_ratio"].append(float(np.mean(motion_mask > 0)))

            prev_gray = gray

        return {
            "flow_frames": flow_frames,
            "mask_frames": mask_frames,
            "metrics": metrics,
        }