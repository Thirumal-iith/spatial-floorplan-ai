"""
benchmark/dataset_generator.py

Builds the SYNTHETIC benchmark set from benchmark/synth_scene.py:
  * ground_truth.json                       - design-spec dimensions (labelled synthetic)
  * tier3_lidar_raw/                        - rendered depth + drifting poses, whole property (perceived by pipeline)
  * tier3_lidar_raw_bedroom_repeat/         - second LiDAR capture of the bedroom (repeatability gate)
  * tier2_video/scenario.json               - geometry passthrough + drifting poses (video perception TODO)
  * tier1_photos/<room>/ + scenario.json    - geometry passthrough, no poses (photo perception TODO)

Usage: python -m benchmark.dataset_generator [--skip-render]
"""

import os
import sys
import json
import copy
import argparse
from typing import Dict, Any

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark import synth_scene as S  # noqa: E402
from pipeline.geometry.drift_correction import se2_of  # noqa: E402


def build_benchmark_dataset(base_dir: str = "benchmark/data", render_lidar: bool = True):
    os.makedirs(base_dir, exist_ok=True)
    gt = S.ground_truth()
    with open(os.path.join(base_dir, "ground_truth.json"), "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=1)

    traj, observe = S.property_trajectory()

    if render_lidar:
        n = S.write_capture(os.path.join(base_dir, "tier3_lidar_raw"), traj, seed=11, odom=S.LIDAR_ODOM,
                            label="whole property, protocol walk")
        print(f"[BENCHMARK] Rendered LiDAR whole-property capture: {n} frames")
        btraj, _ = S.room_trajectory("room_bedroom", scale=0.9)
        n = S.write_capture(os.path.join(base_dir, "tier3_lidar_raw_bedroom_repeat"), btraj, seed=12,
                            odom=S.LIDAR_ODOM, label="bedroom repeat capture")
        print(f"[BENCHMARK] Rendered LiDAR bedroom repeat capture: {n} frames")

    video_dir = os.path.join(base_dir, "tier2_video")
    os.makedirs(video_dir, exist_ok=True)
    _create_scenario_file(video_dir, gt, noise_std=0.025,
                          seed=3, odom=S.VIDEO_ODOM, traj=(traj, observe))

    photos_dir = os.path.join(base_dir, "tier1_photos")
    os.makedirs(photos_dir, exist_ok=True)
    for rid in gt["rooms"]:
        os.makedirs(os.path.join(photos_dir, rid), exist_ok=True)
    _create_scenario_file(photos_dir, gt, noise_std=0.065,
                          seed=4, odom=None, traj=None)

    print(f"[BENCHMARK] Synthetic benchmark dataset written to: {base_dir}")
    return base_dir


def _create_scenario_file(target_dir: str, gt: Dict[str, Any], noise_std: float, seed: int, odom, traj):
    """
    Geometry passthrough for tiers whose perception is not implemented yet.
    With `odom`, rooms are stored in the camera frame of the keyframe they were observed at and
    poses are drifting odometry, so placement still depends on drift correction.
    """
    rng = np.random.default_rng(seed)
    rooms_gt = copy.deepcopy(gt["rooms"])
    rooms, staged, poses = [], [], []
    true_traj, observe = (traj if traj else (None, {}))
    if odom is not None:
        raw = S.simulate_odometry(true_traj, rng, odom)
        poses = [S._pose_zup(x, y, yaw, 0.0, S.CAM_H).round(
            6).tolist() for x, y, yaw in raw]

    for rid, rdata in rooms_gt.items():
        local = np.asarray(rdata["polygon"], dtype=float) + \
            rng.normal(0, noise_std * 0.2, (4, 2))
        room_obj = {"room_id": rid, "name": rdata["name"]}
        if odom is not None and rid in observe:
            k = observe[rid]
            world = local + np.asarray(rdata["world_polygon"][0])
            T = S._pose_zup(*true_traj[k], 0.0, S.CAM_H)
            R2, t2 = se2_of(T)
            cam = (R2.T @ (world - t2).T).T
            room_obj.update({"polygon": np.round(cam, 4).tolist(), "observed_at_pose": int(k),
                             "polygon_frame": "camera"})
        else:
            room_obj.update({"polygon": np.round(
                local, 4).tolist(), "polygon_frame": "room_local"})

        raw_ops = []
        for w_idx, wall in enumerate(rdata["walls"]):
            for op in wall.get("openings", []):
                raw_ops.append({
                    "opening_id": op["opening_id"], "wall_index": w_idx, "type": op["type"],
                    "width_m": round(op["width_m"] + rng.normal(0, noise_std * 0.3), 4),
                    "height_m": op["height_m"], "connected_room_id": op.get("connected_room"),
                })
        room_obj["ceiling_height_m"] = round(
            rdata["ceiling_height_m"] + rng.normal(0, noise_std * 0.2), 4)
        room_obj["openings"] = raw_ops
        rooms.append(room_obj)
        staged.extend(rdata.get("staged_damages", []))

    scenario = {"property_id": gt["property_id"], "data_provenance": "synthetic",
                "rooms": rooms, "staged_damages": staged, "poses": poses}
    with open(os.path.join(target_dir, "scenario.json"), "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-render", action="store_true",
                    help="Do not re-render LiDAR captures")
    build_benchmark_dataset(render_lidar=not ap.parse_args().skip_render)
