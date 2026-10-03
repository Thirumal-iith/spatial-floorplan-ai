"""
pipeline/geometry/photo_layout.py

Photo tier: one folder per room, 2-8 stills taken turning in place from one spot (protocol:
iPhone 0.5x ultra-wide, landscape, phone at chest height, floor-wall line visible).
No poses, no depth: only JPEG + EXIF.

Per room:
  1. Intrinsics from EXIF FocalLengthIn35mmFilm (fx = f35 / 36 mm * image width).
  2. Gravity per photo from the vertical vanishing point (line segments, least squares).
  3. Relative yaw between consecutive photos from an ORB + RANSAC homography (pure rotation:
     H = K R K^-1). The 360 deg loop (last -> first) is closed by spreading the residual.
  4. Pseudo depth per photo with the shared mono-layout code; camera height above the floor is the
     protocol prior (chest height 1.40 m, CI from its +-0.08 m spread). This prior is the main
     reason the photo CI is wider than video's.
  5. LidarReconstructor (no drift correction, Manhattan-aligned output) -> one room polygon,
     ceiling height, openings.

Across rooms: rooms are placed by matching doors (MultiRoomStitcher portal matching).
"""

import glob
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image

from pipeline.geometry.mono_layout import ColorModels, FrameLayout, ray_dirs, _lab
from pipeline.geometry.lidar_layout import LidarReconstructor
from pipeline.geometry.video_layout import strided_K

STRIDE = 3
CAM_HEIGHT_PRIOR_M = 1.40
DEFAULT_F35_MM = 13.0
D_CV = np.diag([1.0, -1.0, -1.0])  # OpenCV camera <-> ARKit camera axes


def list_photos(folder: str) -> List[str]:
    exts = (".jpg", ".jpeg", ".png")
    return sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(exts))


def is_photo_capture(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    subs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]
    return any(len(list_photos(d)) >= 2 for d in subs)


def intrinsics(path: str, W: int, H: int) -> Tuple[np.ndarray, Dict]:
    f35, src = None, "default (13 mm, iPhone 0.5x)"
    try:
        ex = Image.open(path).getexif()
        v = ex.get_ifd(0x8769).get(0xA405) or ex.get(0xA405)
        if v:
            f35, src = float(v), "EXIF FocalLengthIn35mmFilm"
    except Exception:
        pass
    f35 = f35 or DEFAULT_F35_MM
    fx = f35 / 36.0 * max(W, H)
    return np.array([[fx, 0, W / 2.0], [0, fx, H / 2.0], [0, 0, 1.0]]), {"f35_mm": f35, "source": src}


# ------------------------------------------------------------------ gravity
def _segments(gray: np.ndarray) -> np.ndarray:
    try:
        lsd = cv2.createLineSegmentDetector()
        segs = lsd.detect(gray)[0]
        if segs is not None:
            return segs.reshape(-1, 4)
    except Exception:
        pass
    edges = cv2.Canny(gray, 50, 150)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 360, 40,
                           minLineLength=30, maxLineGap=4)
    return np.zeros((0, 4)) if segs is None else segs.reshape(-1, 4).astype(float)


def up_vector(img: np.ndarray, K: np.ndarray) -> Optional[np.ndarray]:
    """Gravity 'up' in ARKit camera coordinates from the vertical vanishing point."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = _segments(gray)
    if len(s) == 0:
        return None
    d = s[:, 2:] - s[:, :2]
    L = np.linalg.norm(d, axis=1)
    # 0 = image-vertical
    ang = np.degrees(np.arctan2(np.abs(d[:, 0]), np.abs(d[:, 1])))
    m = (L > 25) & (ang < 25)
    if m.sum() < 8:
        return None
    Kinv = np.linalg.inv(K)
    p1 = np.c_[s[m, :2], np.ones(m.sum())] @ Kinv.T
    p2 = np.c_[s[m, 2:], np.ones(m.sum())] @ Kinv.T
    lines = np.cross(p1, p2)
    lines /= np.linalg.norm(lines, axis=1, keepdims=True)
    w = L[m]
    v = None
    for _ in range(3):  # IRLS: down-weight segments that are not vertical in 3D
        M = (lines * w[:, None]).T @ lines
        v = np.linalg.eigh(M)[1][:, 0]
        r = np.abs(lines @ v)
        w = L[m] / (1.0 + (r / 0.01) ** 2)
    up_cv = -v if v[1] > 0 else v  # OpenCV y points down; up has negative y
    up = D_CV @ up_cv
    return up / np.linalg.norm(up)


def level_rotation(up_cam: np.ndarray) -> np.ndarray:
    """Camera->world rotation with world z = up and the camera's forward heading = world +x."""
    f = np.array([0.0, 0.0, -1.0])
    h = f - (f @ up_cam) * up_cam
    h /= np.linalg.norm(h)
    a2 = np.cross(up_cam, h)
    # rows = world axes in camera coords -> R (cam->world)
    return np.vstack([h, a2, up_cam])


def _rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ------------------------------------------------------------------ yaw
def relative_yaw(img1, img2, K, R1, R2_level) -> Optional[float]:
    """Yaw of photo 2 relative to photo 1 (radians, CCW about up) from a pure-rotation homography."""
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    try:
        det, norm = cv2.SIFT_create(4000, contrastThreshold=0.01), cv2.NORM_L2
    except AttributeError:
        det, norm = cv2.ORB_create(4000, fastThreshold=5), cv2.NORM_HAMMING
    k1, d1 = det.detectAndCompute(g1, None)
    k2, d2 = det.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 20 or len(k2) < 20:
        return None
    knn = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    mt = [p[0]
          for p in knn if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(mt) < 20:
        return None
    a = np.float32([k1[x.queryIdx].pt for x in mt])
    b = np.float32([k2[x.trainIdx].pt for x in mt])
    H, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
    if H is None or inl.sum() < 25:
        return None
    Rcv = np.linalg.inv(K) @ H @ K
    Rcv /= np.cbrt(np.linalg.det(Rcv))
    U, _, Vt = np.linalg.svd(Rcv)
    R21 = D_CV @ (U @ Vt) @ D_CV          # ARKit cam1 -> cam2
    R_w2 = R1 @ R21.T                      # cam2 -> world
    f2 = R_w2 @ np.array([0, 0, -1.0])
    f1 = R1 @ np.array([0, 0, -1.0])
    return float(np.arctan2(f2[1], f2[0]) - np.arctan2(f1[1], f1[0]))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class PhotoRoomReconstructor:
    def reconstruct_room(self, folder: str, room_id: str) -> Tuple[Optional[Dict], Dict]:
        paths = list_photos(folder)
        imgs = [cv2.imread(p, cv2.IMREAD_COLOR) for p in paths]
        imgs = [i for i in imgs if i is not None]
        tel: Dict = {"photos": len(imgs)}
        if len(imgs) < 2:
            return None, {**tel, "error": "fewer than 2 photos"}
        H, W = imgs[0].shape[:2]
        K, ksrc = intrinsics(paths[0], W, H)
        tel["intrinsics"] = ksrc

        ups = [up_vector(im, K) for im in imgs]
        ups = [u if u is not None else np.array(
            [0.0, np.cos(np.radians(5)), np.sin(np.radians(5))]) for u in ups]
        Rl = [level_rotation(u) for u in ups]

        # chain yaws; close the loop if the last photo overlaps the first
        n = len(imgs)
        steps, ok = [], 0
        for i in range(n - 1):
            y = relative_yaw(imgs[i], imgs[i + 1], K, Rl[i], Rl[i + 1])
            ok += y is not None
            steps.append(_wrap(y) if y is not None else None)
        known = [s for s in steps if s is not None]
        fill = float(np.median(known)) if known else 2 * np.pi / n
        steps = [s if s is not None else fill for s in steps]
        close = relative_yaw(imgs[-1], imgs[0], K, Rl[-1], Rl[0])
        loop_err = None
        if close is not None and n >= 3:
            total = sum(steps) + _wrap(close)
            k = np.round(total / (2 * np.pi))
            loop_err = total - k * 2 * np.pi
            steps = [s - loop_err / n for s in steps]
        yaws = np.concatenate([[0.0], np.cumsum(steps)])
        Rs = [_rz(y) @ R for y, R in zip(yaws, Rl)]
        tel.update({"yaw_links_matched": ok, "yaw_links_total": n - 1,
                    "yaw_loop_residual_deg": None if loop_err is None else round(float(np.degrees(loop_err)), 2)})

        # colour models (per room) and per-photo layouts
        samples = []
        for im, R in zip(imgs, Rs):
            dw, _ = ray_dirs(K, W, H, R, STRIDE)
            samples.append((_lab(im, STRIDE), np.arctan2(
                dw[..., 2], np.linalg.norm(dw[..., :2], axis=-1))))
        models = ColorModels().fit(samples)
        layouts = [FrameLayout(im, K, R, models, STRIDE)
                   for im, R in zip(imgs, Rs)]
        rs = np.array([r for r in (L.ceiling_ratio()
                      for L in layouts) if r is not None])
        if len(rs) == 0:
            return None, {**tel, "error": "ceiling-wall boundary never visible"}
        ratio = float(np.median(rs))
        tilts = []
        for _ in range(2):  # alternate: per-photo tilt correction <-> room-wide ceiling ratio
            tilts = [L.refine_tilt(np.log(ratio)) for L in layouts]
            rs2 = np.array([r for r in (L.ceiling_ratio()
                           for L in layouts) if r is not None])
            if len(rs2):
                ratio = float(np.median(rs2))
        Rs = [L.R for L in layouts]
        tel["tilt_corrections_deg"] = [
            [round(p, 2), round(r, 2)] for p, r in tilts]
        hf = CAM_HEIGHT_PRIOR_M
        hc = ratio * hf
        frames = []
        for i, (L, R) in enumerate(zip(layouts, Rs)):
            T = np.eye(4)
            T[:3, :3] = R
            T[2, 3] = hf
            d = L.pseudo_depth(R, hf, hc)
            frames.append({"index": f"{i:05d}", "pose": T, "K": strided_K(K, STRIDE), "depth": d,
                           "conf": np.full(d.shape, 2, np.uint8)})
        rooms, _, rtel = LidarReconstructor(tier="photos", align_output=True).reconstruct_frames(
            frames, drift_correction=False)
        # the room is the free-space region that contains the camera
        best = None
        for r in rooms:
            P = np.asarray(r["polygon"], np.float32).reshape(-1, 1, 2)
            if cv2.pointPolygonTest(P, (0.0, 0.0), False) >= 0:
                best = r
                break
        if best is None and rooms:
            best = max(rooms, key=lambda r: cv2.contourArea(
                np.asarray(r["polygon"], np.float32)))
        if best is None:
            return None, {**tel, "error": "no room region"}
        tel.update({"camera_height_prior_m": hf, "ceiling_ratio": round(ratio, 4),
                    "ceiling_ratio_frames": int(len(rs)), "segmentation": rtel})
        best = self._rename(best, room_id)
        return best, tel

    @staticmethod
    def _rename(room: Dict, rid: str) -> Dict:
        old = room["room_id"]
        room["room_id"] = rid
        for w in room["walls"]:
            w["wall_id"] = w["wall_id"].replace(old, rid, 1)
            for o in w["openings"]:
                o["opening_id"] = o["opening_id"].replace(old, rid, 1)
                o["connected_room_id"] = None   # unknown until stitched
        room["pose_placed"] = False
        return room


def reconstruct_property(path: str) -> Tuple[List[Dict], Dict]:
    rooms, tel = [], {}
    for d in sorted(glob.glob(os.path.join(path, "*"))):
        if not os.path.isdir(d) or len(list_photos(d)) < 2:
            continue
        rid = os.path.basename(d)
        try:
            r, t = PhotoRoomReconstructor().reconstruct_room(d, rid)
        except Exception as e:  # one bad room must not kill the property
            r, t = None, {"error": f"{type(e).__name__}: {e}"}
        tel[rid] = t
        if r is not None:
            name = rid.replace("_", " ").title()
            r["name"] = name
            r["room_type"] = "hallway" if any(k in rid.lower() for k in (
                "hall", "connector", "corridor")) else r["room_type"]
            rooms.append(r)
    return rooms, tel
