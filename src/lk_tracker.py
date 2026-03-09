import cv2
import numpy as np


class LKTracker:
    def __init__(
        self,
        max_corners: int = 300,
        quality_level: float = 0.01,
        min_distance: int = 7,
        block_size: int = 7,
        win_size: int = 21,
        max_level: int = 3,
    ):
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self.lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def track(self, gray_frames, detect_interval: int = 10):
        prev_gray = gray_frames[0]
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)

        tracks = []
        active_ids = []
        next_track_id = 0

        metrics = {
            "num_points": [],
            "mean_displacement": [],
            "lost_points": [],
        }

        vis_frames = []

        if prev_pts is None:
            prev_pts = np.empty((0, 1, 2), dtype=np.float32)

        for p in prev_pts.reshape(-1, 2):
            tracks.append([tuple(p)])
            active_ids.append(next_track_id)
            next_track_id += 1

        for i in range(1, len(gray_frames)):
            gray = gray_frames[i]
            canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            lost_points = 0
            mean_disp = 0.0

            if len(prev_pts) > 0:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, prev_pts, None, **self.lk_params
                )

                if next_pts is None or status is None:
                    next_pts = np.empty((0, 1, 2), dtype=np.float32)
                    status = np.empty((0, 1), dtype=np.uint8)

                good_new = next_pts[status == 1]
                good_old = prev_pts[status == 1]
                status_flat = status.reshape(-1)

                new_active_ids = []
                displacements = []

                good_idx = 0
                for idx, track_id in enumerate(active_ids):
                    if idx < len(status_flat) and status_flat[idx] == 1:
                        new_point = tuple(good_new[good_idx])
                        old_point = tuple(good_old[good_idx])

                        tracks[track_id].append(new_point)
                        new_active_ids.append(track_id)

                        displacements.append(
                            np.linalg.norm(np.array(new_point) - np.array(old_point))
                        )
                        good_idx += 1
                    else:
                        lost_points += 1

                for track_id in new_active_ids:
                    pts = np.array(tracks[track_id], dtype=np.int32)
                    if len(pts) >= 2:
                        cv2.polylines(canvas, [pts], False, (0, 255, 0), 1)
                    x, y = tracks[track_id][-1]
                    cv2.circle(canvas, (int(x), int(y)), 2, (0, 0, 255), -1)

                prev_pts = good_new.reshape(-1, 1, 2)
                active_ids = new_active_ids

                if displacements:
                    mean_disp = float(np.mean(displacements))

            if i % detect_interval == 0:
                mask = np.full_like(gray, 255)
                for p in prev_pts.reshape(-1, 2):
                    cv2.circle(mask, (int(p[0]), int(p[1])), 10, 0, -1)

                new_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                if new_pts is not None:
                    if len(prev_pts) == 0:
                        prev_pts = new_pts
                    else:
                        prev_pts = np.vstack([prev_pts, new_pts])

                    for p in new_pts.reshape(-1, 2):
                        tracks.append([tuple(p)])
                        active_ids.append(next_track_id)
                        next_track_id += 1

            metrics["num_points"].append(int(len(prev_pts)))
            metrics["mean_displacement"].append(mean_disp)
            metrics["lost_points"].append(lost_points)

            vis_frames.append(canvas)
            prev_gray = gray

        return {
            "tracks": tracks,
            "metrics": metrics,
            "vis_frames": vis_frames,
        }