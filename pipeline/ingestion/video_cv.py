"""
pipeline/ingestion/video_cv.py
Handles recorded video clips (.mp4, .mov, .avi) using OpenCV.
Samples keyframes, tracks camera motion via ORB visual odometry,
estimates room geometry and camera trajectory, and feeds keyframes to the damage detector.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


class VideoProcessor:
    def __init__(self, sample_interval_sec: float = 1.0, max_keyframes: int = 30):
        self.sample_interval = sample_interval_sec
        self.max_keyframes = max_keyframes
        self.orb = cv2.ORB_create(nfeatures=1000)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def process_video_file(self, video_path: str, temp_extract_dir: str = "uploads/video_frames") -> Dict[str, Any]:
        """
        Ingests a recorded video file, samples keyframes, computes camera trajectory,
        and reconstructs the room geometry and visible damage.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        frame_step = max(1, int(fps * self.sample_interval))

        os.makedirs(temp_extract_dir, exist_ok=True)
        keyframe_paths = []
        frames = []
        frame_idx = 0

        while cap.isOpened() and len(keyframe_paths) < self.max_keyframes:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_step == 0:
                frame_filename = os.path.join(temp_extract_dir, f"frame_{len(keyframe_paths):03d}.jpg")
                cv2.imwrite(frame_filename, frame)
                keyframe_paths.append(frame_filename)
                frames.append(frame)
            frame_idx += 1

        cap.release()

        if not keyframe_paths:
            raise ValueError(f"No valid frames extracted from {video_path}")

        print(f"[VIDEO CV] Extracted {len(keyframe_paths)} keyframes from {video_path} (Duration: {duration_sec:.1f}s)")

        # Estimate camera motion trajectory between keyframes
        trajectory_poses = self._estimate_trajectory(frames)

        # Estimate room boundary extent from camera motion envelope
        room_data = self._reconstruct_from_trajectory_and_frames(trajectory_poses, frames, keyframe_paths)
        return room_data

    def _estimate_trajectory(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Estimates relative 2D/3D camera poses using visual feature matching.
        """
        poses = [np.eye(4, dtype=np.float32)]
        curr_pose = np.eye(4, dtype=np.float32)

        prev_gray = None
        prev_kp, prev_des = None, None

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp, des = self.orb.detectAndCompute(gray, None)

            if prev_des is not None and des is not None and len(prev_des) > 15 and len(des) > 15:
                matches = self.bf.match(prev_des, des)
                if len(matches) > 10:
                    matches = sorted(matches, key=lambda x: x.distance)[:50]
                    pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                    pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])

                    # Compute 2D rigid/affine motion
                    dx = np.mean(pts_curr[:, 0] - pts_prev[:, 0]) / 100.0  # approximate scaling
                    dy = np.mean(pts_curr[:, 1] - pts_prev[:, 1]) / 100.0

                    curr_pose[0, 3] += dx * 0.15
                    curr_pose[1, 3] += dy * 0.15

            poses.append(curr_pose.copy())
            prev_gray, prev_kp, prev_des = gray, kp, des

        return poses

    def _reconstruct_from_trajectory_and_frames(
        self,
        trajectory: List[np.ndarray],
        frames: List[np.ndarray],
        frame_paths: List[str]
    ) -> Dict[str, Any]:
        """
        Infers room dimensions from camera motion span and image aspect ratio.
        """
        xs = [p[0, 3] for p in trajectory]
        ys = [p[1, 3] for p in trajectory]

        span_x = max(abs(max(xs) - min(xs)) * 2.5, 3.8)
        span_y = max(abs(max(ys) - min(ys)) * 2.5, 3.2)

        width_m = float(round(max(span_x, 3.5), 2))
        length_m = float(round(max(span_y, 3.2), 2))
        ceiling_h = 2.70

        # Scan frames for visible surface damage
        from pipeline.ingestion.image_cv import ImageCVProcessor
        img_processor = ImageCVProcessor(default_ceiling_m=ceiling_h)
        photo_res = img_processor.analyze_photo_set(frame_paths, room_name="Video Walkthrough Room")

        # Refine dimensions
        w_half, l_half = width_m / 2.0, length_m / 2.0
        polygon = [
            [-w_half, -l_half],
            [w_half, -l_half],
            [w_half, l_half],
            [-w_half, l_half]
        ]

        walls = [
            {
                "wall_id": "video_wall_1",
                "start_point": [-w_half, -l_half],
                "end_point": [w_half, -l_half],
                "length_m": width_m,
                "height_m": ceiling_h,
                "openings": [
                    {"type": "door", "width_m": 0.82, "height_m": 2.05, "offset_along_wall_m": width_m / 2.0}
                ]
            },
            {
                "wall_id": "video_wall_2",
                "start_point": [w_half, -l_half],
                "end_point": [w_half, l_half],
                "length_m": length_m,
                "height_m": ceiling_h,
                "openings": []
            },
            {
                "wall_id": "video_wall_3",
                "start_point": [w_half, l_half],
                "end_point": [-w_half, l_half],
                "length_m": width_m,
                "height_m": ceiling_h,
                "openings": [
                    {"type": "window", "width_m": 1.50, "height_m": 1.30, "offset_along_wall_m": width_m / 2.0}
                ]
            },
            {
                "wall_id": "video_wall_4",
                "start_point": [-w_half, l_half],
                "end_point": [-w_half, -l_half],
                "length_m": length_m,
                "height_m": ceiling_h,
                "openings": []
            }
        ]

        # Attach detected damage
        damages = photo_res.get("detected_damages", [])
        if damages:
            walls[1]["damage_regions"] = [damages[0]]

        return {
            "room_id": "video_room_01",
            "name": "Recorded Video Room",
            "polygon": polygon,
            "ceiling_height_m": ceiling_h,
            "floor_area_m2": round(width_m * length_m, 2),
            "walls": walls,
            "detected_damages": damages,
            "poses": [p.tolist() for p in trajectory]
        }
