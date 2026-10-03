"""
pipeline/ingestion/lidar_capture.py

Loader for the 3D Scanner App (Laan Labs) "All Data" raw export, the stock capture tool named in
app/capture_protocol.md:

    frame_XXXXX.json   {"cameraPoseARFrame": [16 floats, row-major 4x4 camera-to-world, ARKit y-up],
                        "intrinsics": [9 floats, row-major 3x3, for the RGB image], ...}
    depth_XXXXX.png    uint16 depth in millimetres (256x192 on iPhone Pro LiDAR)
    conf_XXXXX.png     uint8 ARKit confidence 0 (low) / 1 (medium) / 2 (high)       [optional]
    frame_XXXXX.jpg    RGB image                                                     [optional]

All poses are converted to a z-up world (x, y on the floor plane, z = height).
Camera frame stays ARKit's: x right, y up, looking down -z.
"""

import os
import glob
import json
from typing import Dict, List, Optional

import numpy as np
import cv2

C_ARKIT_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
DEFAULT_RGB_SIZE = (1920, 1440)


def is_lidar_capture(path: str) -> bool:
    return os.path.isdir(path) and bool(glob.glob(os.path.join(path, "frame_*.json"))) \
        and bool(glob.glob(os.path.join(path, "depth_*.png")))


def capture_info(path: str) -> Dict:
    p = os.path.join(path, "capture_info.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _rgb_width(path: str, idx: str, meta: Dict) -> float:
    if meta.get("rgb_width"):
        return float(meta["rgb_width"])
    jpg = os.path.join(path, f"frame_{idx}.jpg")
    if os.path.exists(jpg):
        img = cv2.imread(jpg, cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if img is not None:
            return float(img.shape[1] * 8)
    return float(DEFAULT_RGB_SIZE[0])


def load_capture(path: str) -> List[Dict]:
    frames = []
    for jf in sorted(glob.glob(os.path.join(path, "frame_*.json"))):
        idx = os.path.basename(jf)[len("frame_"):-len(".json")]
        dp = os.path.join(path, f"depth_{idx}.png")
        if not os.path.exists(dp):
            continue
        with open(jf, encoding="utf-8") as f:
            meta = json.load(f)
        depth = cv2.imread(dp, cv2.IMREAD_UNCHANGED)
        if depth is None:
            continue
        depth = depth.astype(np.float32) / 1000.0
        cp = os.path.join(path, f"conf_{idx}.png")
        conf = cv2.imread(cp, cv2.IMREAD_UNCHANGED) if os.path.exists(cp) else None
        if conf is None:
            conf = np.full(depth.shape, 2, dtype=np.uint8)

        P = np.asarray(meta["cameraPoseARFrame"], dtype=float).reshape(4, 4)
        K = np.asarray(meta["intrinsics"], dtype=float).reshape(3, 3).copy()
        s = depth.shape[1] / _rgb_width(path, idx, meta)
        K[0, 0] *= s; K[1, 1] *= s; K[0, 2] *= s; K[1, 2] *= s

        T = np.eye(4)
        T[:3, :3] = C_ARKIT_TO_ZUP @ P[:3, :3]
        T[:3, 3] = C_ARKIT_TO_ZUP @ P[:3, 3]
        frames.append({"index": idx, "pose": T, "K": K, "depth": depth, "conf": conf})
    return frames


def backproject(frame: Dict, pose: Optional[np.ndarray] = None, stride: int = 2,
                min_conf: int = 2, max_depth: float = 5.0) -> np.ndarray:
    """Depth image -> (N,3) float32 world points (z-up)."""
    d = frame["depth"][::stride, ::stride]
    c = frame["conf"][::stride, ::stride]
    H, W = frame["depth"].shape
    vv, uu = np.mgrid[0:H:stride, 0:W:stride].astype(np.float32) + 0.5
    m = (d > 0.15) & (d < max_depth) & (c >= min_conf)
    K = frame["K"]
    z = d[m]
    pc = np.stack([(uu[m] - K[0, 2]) / K[0, 0] * z, -(vv[m] - K[1, 2]) / K[1, 1] * z, -z], axis=1)
    P = frame["pose"] if pose is None else pose
    return (pc @ P[:3, :3].T + P[:3, 3]).astype(np.float32)
