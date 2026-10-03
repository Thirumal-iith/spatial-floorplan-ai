"""
benchmark/synth_scene.py

SYNTHETIC benchmark property: single source of truth for
  * room geometry and openings (consistent topology),
  * ground truth (interior face-to-face dimensions),
  * capture trajectories that follow app/capture_protocol.md,
  * a ray-cast LiDAR depth renderer that writes the 3D Scanner App "All Data" export layout:
        frame_XXXXX.json  {"cameraPoseARFrame": 16 floats (row-major, ARKit y-up camera-to-world),
                           "intrinsics": 9 floats (for the RGB image), "time": s}
        depth_XXXXX.png   uint16 millimetres, 256x192
        conf_XXXXX.png    uint8 ARKit confidence 0/1/2
    Stored poses are DRIFTING odometry (simulated VIO bias + noise); depth is rendered from the true pose.

Everything here is synthetic and is labelled as such in every artifact. It exists so that the
perception code paths run end to end and can be regression-tested; it is not evidence of
real-world accuracy.
"""

import os
import json
from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

WALL_T = 0.12
CAM_H = 1.40
DEPTH_W, DEPTH_H = 256, 192
RGB_W, RGB_H = 1920, 1440
FX_RGB = 1450.0
HUB = (0.75, 1.0)  # taped start marker in the connector

ROOMS: Dict[str, Dict] = {
    "connector_hall": {"name": "Central Hallway", "rect": (0.0, 0.0, 1.5, 4.2), "ceiling": 2.70},
    "room_living": {"name": "Living Room", "rect": (1.62, 0.0, 6.82, 4.2), "ceiling": 2.74},
    "room_bedroom": {"name": "Primary Bedroom", "rect": (-4.22, 0.0, -0.12, 3.6), "ceiling": 2.68},
    "room_kitchen": {"name": "Kitchen", "rect": (0.0, 4.32, 3.5, 7.52), "ceiling": 2.71},
}
SIDES = ("S", "E", "N", "W")  # wall order w1..w4 == polygon edge order
OUT_NORMAL = {"S": (0.0, -1.0), "E": (1.0, 0.0),
              "N": (0.0, 1.0), "W": (-1.0, 0.0)}

# center = world coordinate along the wall axis (x for S/N walls, y for E/W walls)
OPENINGS: List[Dict] = [
    {"id": "door_hall_living", "type": "door", "faces": [("connector_hall", "E"), ("room_living", "W")],
     "center": 1.00, "width": 0.82, "sill": 0.0, "height": 2.05},
    {"id": "door_hall_kitchen", "type": "door", "faces": [("connector_hall", "N"), ("room_kitchen", "S")],
     "center": 0.75, "width": 0.80, "sill": 0.0, "height": 2.05},
    {"id": "door_hall_bedroom", "type": "door", "faces": [("connector_hall", "W"), ("room_bedroom", "E")],
     "center": 2.50, "width": 0.85, "sill": 0.0, "height": 2.05},
    {"id": "win_living_south", "type": "window", "faces": [("room_living", "S")],
     "center": 4.20, "width": 1.60, "sill": 0.90, "height": 1.35},
    {"id": "win_bedroom_west", "type": "window", "faces": [("room_bedroom", "W")],
     "center": 1.80, "width": 1.40, "sill": 0.90, "height": 1.20},
    {"id": "win_kitchen_north", "type": "window", "faces": [("room_kitchen", "N")],
     "center": 2.20, "width": 1.00, "sill": 1.00, "height": 1.00},
]

LIDAR_ODOM = {"yaw_bias_per_m": 0.004, "yaw_scale_err": 0.006, "trans_scale_err": 0.008,
              "yaw_noise": 0.0005, "trans_noise": 0.002}
VIDEO_ODOM = {"yaw_bias_per_m": 0.008, "yaw_scale_err": 0.012, "trans_scale_err": 0.02,
              "yaw_noise": 0.0015, "trans_noise": 0.006}


# --------------------------------------------------------------------------- geometry
def wall_segment(rid: str, side: str):
    x0, y0, x1, y1 = ROOMS[rid]["rect"]
    return {"S": ((x0, y0), (x1, y0)), "E": ((x1, y0), (x1, y1)),
            "N": ((x1, y1), (x0, y1)), "W": ((x0, y1), (x0, y0))}[side]


def opening_center_world(op: Dict, rid: str, side: str) -> Tuple[float, float]:
    a, _ = wall_segment(rid, side)
    return (op["center"], a[1]) if side in ("S", "N") else (a[0], op["center"])


def room_centroid(rid: str) -> np.ndarray:
    x0, y0, x1, y1 = ROOMS[rid]["rect"]
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def ground_truth() -> Dict:
    rooms = {}
    for rid, r in ROOMS.items():
        x0, y0, x1, y1 = r["rect"]
        world = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        walls = []
        for i, side in enumerate(SIDES):
            a, b = wall_segment(rid, side)
            length = abs(b[0] - a[0]) + abs(b[1] - a[1])
            ops = []
            for op in OPENINGS:
                if (rid, side) not in [tuple(f) for f in op["faces"]]:
                    continue
                other = [f[0] for f in op["faces"] if f[0] != rid]
                c = opening_center_world(op, rid, side)
                ops.append({
                    "opening_id": f"{op['id']}@{rid}", "type": op["type"],
                    "width_m": op["width"], "height_m": op["height"], "sill_m": op["sill"],
                    "center_world": [round(c[0], 3), round(c[1], 3)],
                    "offset_along_wall_m": round(abs(c[0] - a[0]) + abs(c[1] - a[1]), 3),
                    "connected_room": other[0] if other else None,
                })
            walls.append({"wall_id": f"{rid}_w{i + 1}", "side": side, "length_m": round(length, 3),
                          "world_start": list(a), "world_end": list(b), "openings": ops})
        rooms[rid] = {
            "name": r["name"],
            "polygon": [[round(x - x0, 3), round(y - y0, 3)] for x, y in world],
            "world_polygon": world,
            "ceiling_height_m": r["ceiling"],
            "floor_area_m2": round((x1 - x0) * (y1 - y0), 4),
            "walls": walls,
        }
    rooms["room_living"]["staged_damages"] = [
        {"damage_id": "dmg_living_water_01", "room_id": "room_living", "damage_class": "water_stain",
         "wall_id": "room_living_w2", "wall_index": 1, "extent_m2": 1.45,
         "location_on_surface": {"u_min": 0.80, "u_max": 2.60, "v_min": 0.05, "v_max": 0.65},
         "severity": "severe"},
        {"damage_id": "dmg_living_crack_01", "room_id": "room_living", "damage_class": "drywall_crack",
         "wall_id": "room_living_w4", "wall_index": 3, "extent_m2": 0.35,
         "location_on_surface": {"u_min": 0.60, "u_max": 2.30, "v_min": 1.20, "v_max": 2.50},
         "severity": "moderate"},
    ]
    return {
        "property_id": "synthetic_residence_01",
        "data_provenance": "synthetic",
        "ground_truth_instrument": "SYNTHETIC design specification - NOT a physical measurement",
        "description": "3 rooms + connector, staged damage (2 classes) in living room, bedroom captured twice",
        "rooms": rooms,
        "total_ground_truth_footprint_m2": round(sum(r["floor_area_m2"] for r in rooms.values()), 4),
    }


# --------------------------------------------------------------------------- trajectories
def _door_route(rid: str) -> List[np.ndarray]:
    for op in OPENINGS:
        faces = dict(op["faces"])
        if op["type"] == "door" and rid in faces and "connector_hall" in faces:
            side = faces["connector_hall"]
            c = np.array(opening_center_world(op, "connector_hall", side))
            n = np.array(OUT_NORMAL[side])
            return [c - 0.5 * n, c + (0.5 + WALL_T) * n, room_centroid(rid)]
    raise ValueError(f"No hall door for {rid}")


def _build(start, routes, step=0.1, spin_frames=24):
    traj, observe = [], {}
    pos, yaw = np.array(start, dtype=float), 0.0
    traj.append([pos[0], pos[1], yaw])

    def spin():
        nonlocal yaw
        for k in range(1, spin_frames + 1):
            traj.append([pos[0], pos[1], yaw + 2 * np.pi * k / spin_frames])
        yaw += 2 * np.pi

    def go(tgt):
        nonlocal pos, yaw
        d = np.asarray(tgt, dtype=float) - pos
        dist = float(np.linalg.norm(d))
        if dist < 1e-6:
            return
        dy = (np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi
        for k in range(1, 6):
            traj.append([pos[0], pos[1], yaw + dy * k / 5])
        yaw += dy
        n = max(1, int(np.ceil(dist / step)))
        for k in range(1, n + 1):
            p = pos + d * k / n
            traj.append([p[0], p[1], yaw])
        pos = np.asarray(tgt, dtype=float)

    spin()
    for rid, wps in routes:
        for w in wps:
            go(w)
        if rid:
            observe[rid] = len(traj) - 1
        spin()
        for w in reversed(wps[:-1]):
            go(w)
        go(start)
    return np.array(traj), observe


def property_trajectory():
    routes = [(rid, _door_route(rid))
              for rid in ROOMS if rid != "connector_hall"]
    traj, observe = _build(HUB, routes)
    observe["connector_hall"] = 0
    return traj, observe


def room_trajectory(rid: str, scale: float = 1.0):
    c = room_centroid(rid)
    x0, y0, x1, y1 = ROOMS[rid]["rect"]
    hx, hy = (x1 - x0) / 2 - 0.8, (y1 - y0) / 2 - 0.7
    loop = [c + scale * np.array(o)
            for o in ([hx, hy], [-hx, hy], [-hx, -hy], [hx, -hy])]
    traj, _ = _build(c, [(None, loop)])
    return traj, {rid: 0}


def simulate_odometry(true_traj: np.ndarray, rng, odom: Dict[str, float]) -> np.ndarray:
    """Integrate noisy, biased relative motion -> drifting VIO-like (x, y, yaw)."""
    out = [true_traj[0].copy()]
    for i in range(1, len(true_traj)):
        a, b = true_traj[i - 1], true_traj[i]
        r_inv = np.array([[np.cos(a[2]), np.sin(a[2])],
                         [-np.sin(a[2]), np.cos(a[2])]])
        dt_local = r_inv @ (b[:2] - a[:2])
        dyaw = b[2] - a[2]
        step_len = float(np.linalg.norm(dt_local))
        dyaw_m = dyaw * (1 + odom["yaw_scale_err"]) + odom["yaw_bias_per_m"] * step_len \
            + rng.normal(0, odom["yaw_noise"])
        dt_m = dt_local * (1 + odom["trans_scale_err"]) + \
            rng.normal(0, odom["trans_noise"], 2) * (step_len > 0)
        prev = out[-1]
        r = np.array([[np.cos(prev[2]), -np.sin(prev[2])],
                     [np.sin(prev[2]), np.cos(prev[2])]])
        p = prev[:2] + r @ dt_m
        out.append(np.array([p[0], p[1], prev[2] + dyaw_m]))
    return np.array(out)


# --------------------------------------------------------------------------- renderer
C_ARKIT_TO_ZUP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])


def camera_rotation(yaw: float, pitch: float) -> np.ndarray:
    """z-up world; camera axes x=right, y=up, z=back (ARKit camera convention)."""
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch)
                 * np.sin(yaw), np.sin(pitch)])
    r = np.array([np.sin(yaw), -np.cos(yaw), 0.0])
    u = np.cross(-f, r)
    return np.column_stack([r, u, -f])


def _planes():
    planes = []
    for rid, r in ROOMS.items():
        x0, y0, x1, y1 = r["rect"]
        h = r["ceiling"]
        for side in SIDES:
            holes = [(op["center"] - op["width"] / 2, op["center"] + op["width"] / 2,
                      op["sill"], op["sill"] + op["height"])
                     for op in OPENINGS if (rid, side) in [tuple(f) for f in op["faces"]]]
            if side in ("S", "N"):
                planes.append(
                    (1, y0 if side == "S" else y1, x0, x1, 0.0, h, holes))
            else:
                planes.append(
                    (0, x1 if side == "E" else x0, y0, y1, 0.0, h, holes))
        planes.append((2, 0.0, x0, x1, y0, y1, []))
        planes.append((2, h, x0, x1, y0, y1, []))
    return planes


_PLANES = None


def raycast(R: np.ndarray, t: np.ndarray, W: int, H: int, fx: float):
    """Returns per-pixel depth (along -z_cam, inf = no hit), plane index, world ray dirs."""
    global _PLANES
    if _PLANES is None:
        _PLANES = _planes()
    vv, uu = np.mgrid[0:H, 0:W] + 0.5
    dc = np.stack([(uu - W / 2) / fx, -(vv - H / 2) / fx, -
                  np.ones_like(uu)], axis=-1).reshape(-1, 3)
    dw = dc @ R.T
    best = np.full(len(dw), np.inf)
    pid = np.full(len(dw), -1)
    for i, (k, c, la, ha, lb, hb, holes) in enumerate(_PLANES):
        a_ax, b_ax = [ax for ax in (0, 1, 2) if ax != k]
        dk = dw[:, k]
        ok = np.abs(dk) > 1e-9
        tt = np.where(ok, (c - t[k]) / np.where(ok, dk, 1.0), np.inf)
        m = ok & (tt > 0.05) & (tt < best)
        if not m.any():
            continue
        pa = t[a_ax] + tt * dw[:, a_ax]
        pb = t[b_ax] + tt * dw[:, b_ax]
        m &= (pa >= la) & (pa <= ha) & (pb >= lb) & (pb <= hb)
        for h1, h2, h3, h4 in holes:
            m &= ~((pa > h1) & (pa < h2) & (pb > h3) & (pb < h4))
        best[m] = tt[m]
        pid[m] = i
    return best, pid, dw


def _hash01(ix, iy, seed):
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64)
                                            * 19349663) ^ (seed * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float32) / 65535.0


def _texture(u, v):
    """World-anchored multi-scale texture (so features are trackable across views)."""
    out = np.zeros(len(u), np.float32)
    for cell, amp, sd in ((0.15, 10.0, 1), (0.05, 9.0, 2), (0.02, 5.0, 3)):
        out += amp * (_hash01(np.floor(u / cell),
                      np.floor(v / cell), sd) - 0.5) * 2
    return out


def _damage_paint(rid, side, u, v):
    """Returns (stain_mask, crack_mask) for wall-surface coordinates (u along wall from its start, v height)."""
    stain = np.zeros(len(u), bool)
    crack = np.zeros(len(u), bool)
    if rid != "room_living":
        return stain, crack
    # water stain: u 0.80-2.60, v 0.05-0.65 (irregular blotch)
    if side == "E":
        cu, cv_, ru, rv = 1.70, 0.35, 0.90, 0.30
        rr = ((u - cu) / ru) ** 2 + ((v - cv_) / rv) ** 2
        edge = 0.85 + 0.15 * _hash01(np.floor(u / 0.08), np.floor(v / 0.08), 7)
        stain = rr < edge
    if side == "W":   # zig-zag crack: u 0.60-2.30, v 1.20-2.50
        s = np.clip((u - 0.60) / 1.70, 0, 1)
        vc = 1.20 + 1.30 * s + 0.06 * np.sin(s * 40.0)
        crack = (u > 0.60) & (u < 2.30) & (np.abs(v - vc) < 0.008)
    return stain, crack


_META = None


def _plane_meta():
    meta = []
    for rid, r in ROOMS.items():
        for side in SIDES:
            meta.append(("wall", rid, side))
        meta.append(("floor", rid, None))
        meta.append(("ceiling", rid, None))
    return meta


def render_rgb(R: np.ndarray, t: np.ndarray, W: int, H: int, fx: float, rng,
               exposure: float = 1.0, noise_sigma: float = 2.0) -> np.ndarray:
    """Lambert-free flat-shaded RGB render (BGR uint8) with world-anchored texture, sky behind windows
    and the staged damage painted on the living-room walls."""
    global _META
    if _META is None:
        _META = _plane_meta()
    depth, pid, dw = raycast(R, t, W, H, fx)
    img = np.zeros((H * W, 3), np.float32)
    img[:] = (250, 215, 170)  # sky (no hit: looking out of a window)
    hit = pid >= 0
    P = t + np.where(hit, depth, 0)[:, None] * dw
    for i, (kind, rid, side) in enumerate(_META):
        m = pid == i
        if not m.any():
            continue
        p = P[m]
        if kind == "floor":
            base, tex = np.array(
                [60, 100, 150], np.float32), _texture(p[:, 0], p[:, 1])
        elif kind == "ceiling":
            base, tex = np.array(
                [236, 238, 240], np.float32), 0.3 * _texture(p[:, 0], p[:, 1])
        else:
            a, _ = wall_segment(rid, side)
            if side in ("S", "N"):
                u = np.abs(p[:, 0] - a[0])
                shade = 1.0
            else:
                u = np.abs(p[:, 1] - a[1])
                shade = 0.92
            base = np.array([190, 200, 205], np.float32) * shade
            tex = _texture(u + 10 * SIDES.index(side), p[:, 2])
            stain, crack = _damage_paint(rid, side, u, p[:, 2])
        col = base[None, :] + tex[:, None]
        if kind == "wall":
            col[stain] = col[stain] * 0.55 + \
                np.array([70, 120, 150], np.float32) * 0.45
            col[crack] = (45, 45, 50)
        img[m] = col
    img = img * exposure + rng.normal(0, noise_sigma, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8).reshape(H, W, 3)


def render_depth(R: np.ndarray, t: np.ndarray, rng) -> Tuple[np.ndarray, np.ndarray]:
    global _PLANES
    if _PLANES is None:
        _PLANES = _planes()
    s = DEPTH_W / RGB_W
    fx = fy = FX_RGB * s
    cx, cy = DEPTH_W / 2, DEPTH_H / 2
    vv, uu = np.mgrid[0:DEPTH_H, 0:DEPTH_W] + 0.5
    dc = np.stack([(uu - cx) / fx, -(vv - cy) / fy, -
                  np.ones_like(uu)], axis=-1).reshape(-1, 3)
    dw = dc @ R.T
    best = np.full(len(dw), np.inf)
    cosang = np.zeros(len(dw))
    norms = np.linalg.norm(dw, axis=1)
    for k, c, la, ha, lb, hb, holes in _PLANES:
        a_ax, b_ax = [ax for ax in (0, 1, 2) if ax != k]
        dk = dw[:, k]
        ok = np.abs(dk) > 1e-9
        tt = np.where(ok, (c - t[k]) / np.where(ok, dk, 1.0), np.inf)
        m = ok & (tt > 0.05) & (tt < best)
        if not m.any():
            continue
        pa = t[a_ax] + tt * dw[:, a_ax]
        pb = t[b_ax] + tt * dw[:, b_ax]
        m &= (pa >= la) & (pa <= ha) & (pb >= lb) & (pb <= hb)
        for h1, h2, h3, h4 in holes:
            m &= ~((pa > h1) & (pa < h2) & (pb > h3) & (pb < h4))
        best[m] = tt[m]
        cosang[m] = np.abs(dk[m]) / norms[m]
    valid = np.isfinite(best)
    depth = np.where(valid, best, 0.0)
    depth = depth + valid * \
        rng.normal(0, 1, len(depth)) * (0.0015 + 0.0025 * depth)
    conf = np.where(valid & (cosang > 0.25) & (depth < 4.5),
                    2, np.where(valid & (depth < 5.0), 1, 0))
    return depth.reshape(DEPTH_H, DEPTH_W), conf.reshape(DEPTH_H, DEPTH_W).astype(np.uint8)


def _pose_zup(x, y, yaw, pitch, z):
    T = np.eye(4)
    T[:3, :3] = camera_rotation(yaw, pitch)
    T[:3, 3] = [x, y, z]
    return T


def _to_arkit(T_zup: np.ndarray) -> np.ndarray:
    C = C_ARKIT_TO_ZUP
    T = np.eye(4)
    T[:3, :3] = C.T @ T_zup[:3, :3]
    T[:3, 3] = C.T @ T_zup[:3, 3]
    return T


def write_capture(out_dir: str, true_traj: np.ndarray, seed: int, odom: Dict[str, float],
                  frame_stride: int = 2, label: str = "") -> int:
    """Render a capture in 3D Scanner App 'All Data' layout. Returns number of frames."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.startswith(("frame_", "depth_", "conf_")):
            os.remove(os.path.join(out_dir, f))
    rng = np.random.default_rng(seed)
    odo = simulate_odometry(true_traj, rng, odom)
    n = len(true_traj)
    pitch = np.radians(28.0) * np.sin(2 * np.pi * np.arange(n) / 9.0)
    height = CAM_H + 0.03 * np.sin(np.arange(n) / 7.0)
    K_rgb = [FX_RGB, 0.0, RGB_W / 2, 0.0, FX_RGB, RGB_H / 2, 0.0, 0.0, 1.0]
    count = 0
    for i in range(0, n, frame_stride):
        x, y, yaw = true_traj[i]
        T_true = _pose_zup(x, y, yaw, pitch[i], height[i])
        depth, conf = render_depth(T_true[:3, :3], T_true[:3, 3], rng)
        xo, yo, yawo = odo[i]
        T_odo = _to_arkit(_pose_zup(xo, yo, yawo, pitch[i], height[i]))
        name = f"{count:05d}"
        cv2.imwrite(os.path.join(out_dir, f"depth_{name}.png"),
                    np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16))
        cv2.imwrite(os.path.join(out_dir, f"conf_{name}.png"), conf)
        with open(os.path.join(out_dir, f"frame_{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"cameraPoseARFrame": T_odo.flatten().round(6).tolist(), "intrinsics": K_rgb,
                       "rgb_width": RGB_W, "rgb_height": RGB_H, "time": round(i / 10.0, 3),
                       "frame_index": i}, f)
        count += 1
    with open(os.path.join(out_dir, "capture_info.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "synthetic_renderer (benchmark/synth_scene.py)", "label": label,
                   "seed": seed, "frames": count, "odometry_model": odom,
                   "layout": "3D Scanner App 'All Data' export"}, f, indent=1)
    return count


VIDEO_W, VIDEO_H = 480, 360


def write_video_capture(out_dir: str, true_traj: np.ndarray, seed: int, frame_stride: int = 2,
                        label: str = "", exposure: float = 1.0) -> int:
    """RGB-only walkthrough with VIO poses (no depth): frame_XXXXX.jpg + frame_XXXXX.json.
    Same layout as an ARKit RGB+pose recorder export (e.g. 3D Scanner App on a non-LiDAR iPhone)."""
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.startswith(("frame_", "depth_", "conf_")):
            os.remove(os.path.join(out_dir, f))
    rng = np.random.default_rng(seed)
    odo = simulate_odometry(true_traj, rng, VIDEO_ODOM)
    n = len(true_traj)
    pitch = np.radians(22.0) * np.sin(2 * np.pi * np.arange(n) / 9.0)
    height = CAM_H + 0.03 * np.sin(np.arange(n) / 7.0)
    fx = FX_RGB * VIDEO_W / RGB_W
    K = [fx, 0.0, VIDEO_W / 2, 0.0, fx, VIDEO_H / 2, 0.0, 0.0, 1.0]
    count = 0
    for i in range(0, n, frame_stride):
        x, y, yaw = true_traj[i]
        T_true = _pose_zup(x, y, yaw, pitch[i], height[i])
        img = render_rgb(T_true[:3, :3], T_true[:3, 3],
                         VIDEO_W, VIDEO_H, fx, rng, exposure=exposure)
        xo, yo, yawo = odo[i]
        T_odo = _to_arkit(_pose_zup(xo, yo, yawo, pitch[i], height[i]))
        name = f"{count:05d}"
        cv2.imwrite(os.path.join(out_dir, f"frame_{name}.jpg"), img, [
                    cv2.IMWRITE_JPEG_QUALITY, 92])
        with open(os.path.join(out_dir, f"frame_{name}.json"), "w", encoding="utf-8") as f:
            json.dump({"cameraPoseARFrame": T_odo.flatten().round(6).tolist(), "intrinsics": K,
                       "rgb_width": VIDEO_W, "rgb_height": VIDEO_H, "time": round(i / 10.0, 3),
                       "frame_index": i}, f)
        count += 1
    with open(os.path.join(out_dir, "capture_info.json"), "w", encoding="utf-8") as f:
        json.dump({"source": "synthetic_renderer (benchmark/synth_scene.py)", "label": label,
                   "seed": seed, "frames": count, "odometry_model": VIDEO_ODOM,
                   "layout": "RGB + ARKit pose export (no depth)"}, f, indent=1)
    return count


PHOTO_W, PHOTO_H = 960, 720
# iPhone 0.5x ultra-wide (13 mm equivalent); protocol asks for 0.5x
PHOTO_F35_MM = 13.0


def photo_station(rid: str) -> Tuple[float, float]:
    """Where the protocol tells the user to stand: room centre (offset a little, people are imprecise)."""
    c = room_centroid(rid)
    off = {"connector_hall": (0.05, -0.6), "room_living": (0.25, -0.15),
           "room_bedroom": (-0.2, 0.1), "room_kitchen": (0.15, 0.2)}[rid]
    return float(c[0] + off[0]), float(c[1] + off[1])


def write_photo_capture(out_root: str, seed: int, n_per_room: int = 8) -> Dict[str, int]:
    """Tier 1: one folder per room, n stills taken turning in place at chest height.
    Only JPEGs with standard EXIF (FocalLengthIn35mmFilm) are written - no poses, no depth."""
    from PIL import Image
    rng = np.random.default_rng(seed)
    fx = PHOTO_F35_MM / 36.0 * PHOTO_W
    f35 = int(round(PHOTO_F35_MM))
    counts = {}
    for rid in ROOMS:
        d = os.path.join(out_root, rid)
        os.makedirs(d, exist_ok=True)
        for f in os.listdir(d):
            if f.lower().endswith((".jpg", ".png")):
                os.remove(os.path.join(d, f))
        x, y = photo_station(rid)
        yaw0 = rng.uniform(0, 2 * np.pi)
        for k in range(n_per_room):
            yaw = yaw0 + 2 * np.pi * k / n_per_room + \
                rng.normal(0, np.radians(2))
            pitch = rng.normal(np.radians(-5), np.radians(2.5))
            h = CAM_H + rng.normal(0, 0.03)
            T = _pose_zup(x, y, yaw, pitch, h)
            R = T[:3, :3]
            roll = rng.normal(0, np.radians(1.5))
            Rr = np.array([[np.cos(roll), -np.sin(roll), 0],
                          [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
            img = render_rgb(R @ Rr, T[:3, 3], PHOTO_W, PHOTO_H, fx, rng)
            pil = Image.fromarray(img[:, :, ::-1])
            exif = Image.Exif()
            exif[0x010F] = "SYNTHETIC"          # Make
            exif[0x0110] = "synth_scene render"  # Model
            ifd = exif.get_ifd(0x8769)
            ifd[0xA405] = f35                   # FocalLengthIn35mmFilm
            pil.save(os.path.join(
                d, f"IMG_{k + 1:04d}.jpg"), quality=92, exif=exif)
        counts[rid] = n_per_room
    return counts
