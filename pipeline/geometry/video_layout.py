"""
pipeline/geometry/video_layout.py

Video tier: RGB frames + VIO poses (no depth) -> pseudo depth (mono_layout) -> the same
room segmentation / wall snapping / openings / pose-graph code as the LiDAR tier.

Input layout (ARKit RGB+pose recorder, e.g. 3D Scanner App export on a non-LiDAR iPhone):
    frame_XXXXX.jpg   RGB
    frame_XXXXX.json  {"cameraPoseARFrame": 16 floats, "intrinsics": 9 floats, ...}
"""

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import cv2

from pipeline.ingestion.lidar_capture import C_ARKIT_TO_ZUP
from pipeline.geometry.mono_layout import ColorModels, FrameLayout, estimate_floor_height, ray_dirs, _lab
from pipeline.geometry.lidar_layout import LidarReconstructor
from pipeline.geometry.manhattan import dominant_angle
from pipeline.geometry.drift_correction import PoseGraphOptimizer, _to_se2

STRIDE = 3


def is_video_capture(path: str) -> bool:
    if not os.path.isdir(path) or glob.glob(os.path.join(path, "depth_*.png")):
        return False
    return bool(glob.glob(os.path.join(path, "frame_*.json"))) and bool(glob.glob(os.path.join(path, "frame_*.jpg")))


def load_rgb_capture(path: str) -> List[Dict]:
    frames = []
    for jf in sorted(glob.glob(os.path.join(path, "frame_*.json"))):
        idx = os.path.basename(jf)[len("frame_"):-len(".json")]
        ip = os.path.join(path, f"frame_{idx}.jpg")
        img = cv2.imread(ip, cv2.IMREAD_COLOR)
        if img is None:
            continue
        with open(jf, encoding="utf-8") as f:
            meta = json.load(f)
        P = np.asarray(meta["cameraPoseARFrame"], float).reshape(4, 4)
        K = np.asarray(meta["intrinsics"], float).reshape(3, 3).copy()
        s = img.shape[1] / float(meta.get("rgb_width") or img.shape[1])
        K[:2] *= s
        T = np.eye(4)
        T[:3, :3] = C_ARKIT_TO_ZUP @ P[:3, :3]
        T[:3, 3] = C_ARKIT_TO_ZUP @ P[:3, 3]
        frames.append({"index": idx, "pose": T, "K": K, "img": img})
    return frames


def strided_K(K: np.ndarray, s: int) -> np.ndarray:
    """Intrinsics for the strided grid sampled by ray_dirs (original pixel j*s + 0.5)."""
    Ks = K.copy()
    Ks[0, 0] /= s
    Ks[1, 1] /= s
    Ks[0, 2] = (K[0, 2] - 0.5) / s + 0.5
    Ks[1, 2] = (K[1, 2] - 0.5) / s + 0.5
    return Ks


def fit_color_models(frames: List[Dict], step: int = 3) -> ColorModels:
    samples = []
    for f in frames[::step]:
        H, W = f["img"].shape[:2]
        dw, _ = ray_dirs(f["K"], W, H, f["pose"][:3, :3], STRIDE)
        el = np.arctan2(dw[..., 2], np.linalg.norm(dw[..., :2], axis=-1))
        samples.append((_lab(f["img"], STRIDE), el))
    return ColorModels().fit(samples)


def _smooth(vals: List, k: int = 7) -> np.ndarray:
    v = np.array([np.nan if x is None else x for x in vals], float)
    out = np.empty_like(v)
    for i in range(len(v)):
        w = v[max(0, i - k): i + k + 1]
        w = w[np.isfinite(w)]
        out[i] = np.median(w) if len(w) else np.nan
    if np.isnan(out).all():
        return np.full(len(v), np.nan)
    good = np.where(np.isfinite(out))[0]
    for i in np.where(~np.isfinite(out))[0]:
        out[i] = out[good[np.argmin(np.abs(good - i))]]
    return out


class VideoReconstructor:
    def reconstruct(self, path: str, drift_correction: bool = True) -> Tuple[List[Dict], Dict, Dict]:
        frames = load_rgb_capture(path)
        if len(frames) < 10:
            raise ValueError(
                f"video capture at {path} has only {len(frames)} frames")
        models = fit_color_models(frames)
        layouts = [FrameLayout(f["img"], f["K"], f["pose"]
                               [:3, :3], models, STRIDE) for f in frames]
        poses = [f["pose"] for f in frames]

        # floor height (world z) from multi-view consistency of floor-wall boundaries.
        # Yaw drift blurs that consistency, so it is measured on drift-corrected poses
        # (Manhattan anchors from the boundary directions are scale-free).
        zc = np.array([p[2, 3] for p in poses])
        theta, _ = _to_se2(poses)
        anchors = []
        for k, L in enumerate(layouts):
            if L.f_ok.sum() >= 40:
                a, r1 = L.f_az[L.f_ok], L.f_r1[L.f_ok]
                ang = dominant_angle(np.column_stack(
                    [np.cos(a) * r1, np.sin(a) * r1]))
                if ang is not None:
                    anchors.append((k, ang - theta[k]))
        pre, _ = PoseGraphOptimizer().optimize_trajectory(
            poses, apply_drift_correction=drift_correction, manhattan_anchors=anchors)
        h_med, scale_meta = estimate_floor_height(layouts, pre)
        floor_z = float(np.median(zc) - h_med)
        hf = zc - floor_z
        # Ceiling: only a minority of frames see floor AND ceiling boundaries on the same wall, and
        # single-frame ratios have outliers (doorways). A per-frame value would push ceiling points
        # into the wall band, so one robust capture-wide ratio is used (rooms then share one ceiling
        # height in this tier; the video CI (+-3.5 cm) is set accordingly).
        rs = np.array([r for r in (L.ceiling_ratio()
                      for L in layouts) if r is not None])
        if len(rs) == 0:
            raise ValueError(
                "no frame shows both floor and ceiling boundaries; cannot scale the ceiling")
        med = np.median(rs)
        mad = np.median(np.abs(rs - med)) + 1e-6
        ratio_g = float(np.median(rs[np.abs(rs - med) < 3 * 1.4826 * mad]))
        ratio = np.full(len(layouts), ratio_g)
        hc = ratio * hf

        out_frames = []
        for f, L, a, b in zip(frames, layouts, hf, hc):
            if not np.isfinite(b):
                continue
            d = L.pseudo_depth(f["pose"][:3, :3], float(a), float(b))
            out_frames.append({"index": f["index"], "pose": f["pose"], "K": strided_K(f["K"], STRIDE),
                               "depth": d, "conf": np.full(d.shape, 2, np.uint8)})
        rooms, drift_meta, tel = LidarReconstructor(tier="video").reconstruct_frames(
            out_frames, drift_correction, anchors=anchors if len(out_frames) == len(frames) else None)
        tel.update({"perception": "mono layout (colour-bootstrapped floor/wall/ceiling + gravity) -> pseudo depth",
                    "camera_height_above_floor_m": round(float(np.median(hf)), 3),
                    "scale_from": scale_meta, "colour_models_lab": {k: [round(float(x), 1) for x in v]
                                                                    for k, v in models.centers.items()}})
        return rooms, drift_meta, tel

    @staticmethod
    def _level(p: np.ndarray, dz: float) -> np.ndarray:
        q = p.copy()
        q[2, 3] = dz
        return q
