"""
pipeline/geometry/mono_layout.py

Depth-free layout perception for the Video and Photo tiers (no LiDAR).

Idea (classical single-view layout geometry, as in Delage et al. 2006 / Hedau et al. 2009, made
metric with gravity and one height):
  * Gravity is known (ARKit pose for video, vertical vanishing point for photos), so every pixel ray
    has an elevation angle.
  * Pixels are labelled floor / wall / ceiling / other with per-capture colour models that are
    bootstrapped from geometry: rays pointing steeply down are almost always floor, steeply up are
    ceiling, near-horizontal are mostly wall. No learned model, no download.
  * The floor-wall boundary at azimuth az lies on the floor plane, so its horizontal range is
        D_f(az) = h_cam_above_floor * |ray_xy| / -ray_z.
    A vertical wall is hit at the same horizontal range by every ray with that azimuth, so all wall
    pixels at az get depth from D(az). The ceiling-wall boundary gives D_c(az) = h_ceil_above_cam * ...
  * Where D_f and D_c agree the column is one wall. Where D_f >> D_c the floor continues through a
    doorway under a header: only the floor band (from D_f) and the header band (from D_c) are
    emitted, leaving the doorway as a void in the wall face. Window glass/sky and stains are "other"
    and stay void (the opening plausibility filter rejects low floor-level voids).
  * The ceiling height above the camera comes from the ratio D_f / D_c over agreeing azimuths.
  * The result is a pseudo depth map per frame in exactly the format the LiDAR tier consumes, so the
    same room segmentation, wall snapping, opening and drift code runs on it.

Scale: video uses VIO metric translation and searches the floor height that makes floor-boundary
points from different camera positions agree. Photos have no translation, so the camera height above
the floor is a stated prior (protocol: chest height, 1.40 m +- 0.08 m); this prior dominates the
photo-tier CI.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from pipeline.geometry.manhattan import dominant_angle

FLOOR, WALL, CEIL, OTHER = 0, 1, 2, 3
AZ_BIN = np.radians(0.4)


def _rx(a: float) -> np.ndarray:
    """Rotation about the camera x (right) axis: pitch."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rzc(a: float) -> np.ndarray:
    """Rotation about the camera z (optical) axis: roll."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def ray_dirs(K: np.ndarray, W: int, H: int, R: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray]:
    vv, uu = np.mgrid[0:H:stride, 0:W:stride].astype(np.float64) + 0.5
    dc = np.stack([(uu - K[0, 2]) / K[0, 0], -(vv - K[1, 2]) /
                  K[1, 1], -np.ones_like(uu)], axis=-1)
    dw = dc @ R.T
    return dw, dc


def _lab(img: np.ndarray, stride: int) -> np.ndarray:
    small = cv2.medianBlur(img, 5)[::stride, ::stride]
    return cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)


class ColorModels:
    """Per-capture floor / wall / ceiling colour models bootstrapped from ray elevation."""

    def __init__(self):
        self.centers: Dict[int, np.ndarray] = {}
        self.spread: Dict[int, float] = {}

    def fit(self, samples: List[Tuple[np.ndarray, np.ndarray]]):
        buckets = {FLOOR: [], WALL: [], CEIL: []}
        for lab, el in samples:
            buckets[FLOOR].append(lab[el < np.radians(-32)])
            buckets[CEIL].append(lab[el > np.radians(32)])
            buckets[WALL].append(lab[np.abs(el) < np.radians(6)])
        for k, v in buckets.items():
            v = np.vstack([x for x in v if len(x)]) if any(len(x)
                                                           for x in v) else np.zeros((0, 3))
            if len(v) < 100:
                raise ValueError(
                    f"not enough pixels to bootstrap colour model {k}")
            c = np.median(v, axis=0)
            # second pass: keep the dominant mode only (doorways / windows pollute the wall sample)
            d = np.linalg.norm(v - c, axis=1)
            v2 = v[d < np.percentile(d, 60)]
            c = np.median(v2, axis=0)
            self.centers[k] = c
            self.spread[k] = float(np.percentile(
                np.linalg.norm(v2 - c, axis=1), 90)) + 4.0
        return self

    def classify(self, lab: np.ndarray, el: np.ndarray) -> np.ndarray:
        ks = [FLOOR, WALL, CEIL]
        d = np.stack([np.linalg.norm(lab - self.centers[k],
                     axis=-1) / self.spread[k] for k in ks], axis=-1)
        lbl = np.array(ks)[np.argmin(d, axis=-1)]
        lbl[np.min(d, axis=-1) > 2.2] = OTHER
        # gravity sanity: floor is below the horizon, ceiling above it
        lbl[(lbl == FLOOR) & (el > -np.radians(1))] = OTHER
        lbl[(lbl == CEIL) & (el < np.radians(1))] = OTHER
        return lbl


def _clean(lbl: np.ndarray) -> np.ndarray:
    out = lbl.copy()
    for k in (FLOOR, CEIL, WALL):
        m = (lbl == k).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        out[(lbl == k) & (m == 0)] = OTHER
    return out


def _runs(lbl: np.ndarray):
    """Bottom-connected floor run and top-connected ceiling run per column."""
    H, W = lbl.shape
    floor_top = np.full(W, -1)
    ceil_bot = np.full(W, -1)
    for c in range(W):
        col = lbl[:, c]
        if col[-1] == FLOOR:
            r = H - 1
            while r > 0 and col[r - 1] == FLOOR:
                r -= 1
            floor_top[c] = r
        if col[0] == CEIL:
            r = 0
            while r < H - 1 and col[r + 1] == CEIL:
                r += 1
            ceil_bot[c] = r
    return floor_top, ceil_bot


def _az_profile(az: np.ndarray, D: np.ndarray) -> Dict[int, float]:
    out: Dict[int, List[float]] = {}
    for a, d in zip(az, D):
        out.setdefault(int(np.floor(a / AZ_BIN)), []).append(d)
    return {k: float(np.median(v)) for k, v in out.items()}


class FrameLayout:
    """Per-frame boundary extraction (independent of the floor height, which is applied later)."""

    def __init__(self, img: np.ndarray, K: np.ndarray, R: np.ndarray, models: ColorModels, stride: int):
        H, W = img.shape[:2]
        self.stride, self.K, self.W, self.H = stride, K, W, H
        self.dw, _ = ray_dirs(K, W, H, R, stride)
        nxy = np.linalg.norm(self.dw[..., :2], axis=-1)
        el = np.arctan2(self.dw[..., 2], nxy)
        lab = _lab(img, stride)
        self.lbl = _clean(models.classify(lab, el))
        self.ft, self.cb = _runs(self.lbl)
        self.set_rotation(R)

    def _bdir(self, rows_s, cols_s, sign, R):
        # the boundary lies half a (strided) pixel beyond the last floor/ceiling sample; using the
        # sample centre itself biases every range short by ~1 % at 2 m.
        K, s = self.K, self.stride
        u = cols_s * s + 0.5
        v = rows_s * s + 0.5 + sign * s / 2.0
        dc = np.stack([(u - K[0, 2]) / K[0, 0], -(v - K[1, 2]) /
                      K[1, 1], -np.ones_like(u)], axis=-1)
        d = dc @ R.T
        return np.arctan2(d[:, 1], d[:, 0]), np.arctan2(d[:, 2], np.linalg.norm(d[:, :2], axis=-1))

    def _boundaries(self, R):
        cols = np.arange(self.lbl.shape[1])
        okf = self.ft > 0
        okc = (self.cb >= 0) & (self.cb < self.lbl.shape[0] - 1)
        f_az, f_el = self._bdir(self.ft[okf].astype(
            float), cols[okf].astype(float), -1.0, R)
        c_az, c_el = self._bdir(self.cb[okc].astype(
            float), cols[okc].astype(float), +1.0, R)
        return f_az, f_el, c_az, c_el

    def set_rotation(self, R: np.ndarray):
        self.R = R
        self.dw, _ = ray_dirs(self.K, self.W, self.H, R, self.stride)
        nxy = np.linalg.norm(self.dw[..., :2], axis=-1)
        self.az = np.arctan2(self.dw[..., 1], self.dw[..., 0])
        self.el = np.arctan2(self.dw[..., 2], nxy)
        f_az, f_el, c_az, c_el = self._boundaries(R)
        # unit-height ranges: D = h * cot(|el|)
        self.f_az = f_az
        self.f_r1 = 1.0 / np.tan(-f_el).clip(1e-3)
        self.c_az = c_az
        self.c_r1 = 1.0 / np.tan(c_el).clip(1e-3)
        self.f_ok = np.tan(-f_el) > 0.02
        self.c_ok = np.tan(c_el) > 0.02

    def log_ratios(self, R: np.ndarray) -> np.ndarray:
        """log(D_f/D_c) per common azimuth bin for a candidate rotation (unit heights)."""
        f_az, f_el, c_az, c_el = self._boundaries(R)
        fo, co = np.tan(-f_el) > 0.02, np.tan(c_el) > 0.02
        fp = _az_profile(f_az[fo], 1.0 / np.tan(-f_el[fo]))
        cp = _az_profile(c_az[co], 1.0 / np.tan(c_el[co]))
        return np.array([np.log(fp[k] / cp[k]) for k in fp if k in cp])

    def refine_tilt(self, log_ratio: float, max_deg: float = 3.0) -> Tuple[float, float]:
        """Photo tier: the vanishing-point gravity is only good to ~1-2 deg, which skews floor vs
        ceiling ranges by several %. Search the small pitch/roll correction that makes this frame's
        floor and ceiling boundaries agree with the room-wide ceiling/floor height ratio
        (robust: median absolute deviation over azimuth bins). Returns (pitch, roll) in degrees."""
        best, best_c = (0.0, 0.0), np.inf

        def search(ps, rs):
            nonlocal best, best_c
            for p in ps:
                for r in rs:
                    lr = self.log_ratios(
                        self.R @ _rx(np.radians(p)) @ _rzc(np.radians(r)))
                    if len(lr) < 8:
                        continue
                    c = float(np.median(np.abs(lr - log_ratio)))
                    if c < best_c:
                        best, best_c = (float(p), float(r)), c

        search(np.arange(-max_deg, max_deg + 1e-9, 0.5),
               np.arange(-max_deg, max_deg + 1e-9, 1.0))
        p0, r0 = best
        search(np.arange(p0 - 0.4, p0 + 0.41, 0.1),
               np.arange(r0 - 0.8, r0 + 0.81, 0.4))
        if np.isfinite(best_c):
            self.set_rotation(
                self.R @ _rx(np.radians(best[0])) @ _rzc(np.radians(best[1])))
        return best

    def ceiling_ratio(self) -> Optional[float]:
        """h_ceil_above_cam / h_cam_above_floor from azimuths where both boundaries see one wall."""
        fp = _az_profile(self.f_az[self.f_ok], self.f_r1[self.f_ok])
        cp = _az_profile(self.c_az[self.c_ok], self.c_r1[self.c_ok])
        common = [k for k in fp if k in cp]
        if len(common) < 8:
            return None
        r = np.array([fp[k] / cp[k] for k in common])
        # door columns give larger ratios, a far wall seen over a near ceiling edge smaller ones:
        # take the densest value (histogram mode, refined by the median of its neighbourhood)
        h, e = np.histogram(r, bins=np.arange(r.min(), r.max() + 0.02, 0.01))
        c = 0.5 * (e[int(np.argmax(h))] + e[int(np.argmax(h)) + 1])
        near = r[np.abs(r - c) < 0.03]
        if len(near) < 5:
            return None
        return float(np.median(near))

    def pseudo_depth(self, R: np.ndarray, hf: float, hc: float) -> np.ndarray:
        """Depth along -z_cam per (strided) pixel; 0 where unknown."""
        Hs, Ws = self.lbl.shape
        nxy = np.linalg.norm(self.dw[..., :2], axis=-1).clip(1e-6)
        dz = self.dw[..., 2]
        rng = np.zeros((Hs, Ws))  # horizontal range per pixel
        fl = np.zeros((Hs, Ws), bool)
        ce = np.zeros((Hs, Ws), bool)
        rows = np.arange(Hs)[:, None]
        fl = (self.ft[None, :] >= 0) & (rows >= self.ft[None, :])
        ce = (self.cb[None, :] >= 0) & (rows <= self.cb[None, :])
        rng[fl] = hf * nxy[fl] / np.maximum(-dz[fl], 1e-6)
        rng[ce] = hc * nxy[ce] / np.maximum(dz[ce], 1e-6)
        fl &= dz < -1e-3
        ce &= dz > 1e-3

        fp = _az_profile(self.f_az[self.f_ok], hf * self.f_r1[self.f_ok])
        cp = _az_profile(self.c_az[self.c_ok], hc * self.c_r1[self.c_ok])
        wall = (self.lbl == WALL) & ~fl & ~ce
        keys = np.floor(self.az / AZ_BIN).astype(int)
        Df = np.vectorize(lambda k: fp.get(k, np.nan), otypes=[float])(keys)
        Dc = np.vectorize(lambda k: cp.get(k, np.nan), otypes=[float])(keys)
        zr = dz / nxy  # height gain per metre of horizontal range
        agree = np.isfinite(Df) & np.isfinite(Dc) & (
            np.abs(Df - Dc) < 0.05 * np.minimum(Df, Dc) + 0.03)
        D = np.where(agree, 0.5 * (Df + Dc), np.nan)
        only_f = np.isfinite(Df) & ~np.isfinite(Dc)
        only_c = np.isfinite(Dc) & ~np.isfinite(Df)
        D = np.where(only_f, Df, D)
        D = np.where(only_c, Dc, D)
        split = np.isfinite(Df) & np.isfinite(Dc) & ~agree
        # split columns: floor band from D_f (<= 0.25 m above floor), header band from D_c (>= 2.12 m)
        # height above floor if the pixel lies at D_f
        zf = hf + zr * np.nan_to_num(Df)
        zc = hf + zr * np.nan_to_num(Dc)
        D = np.where(split & (zf <= 0.25), Df, D)
        D = np.where(split & (zc >= 2.12) & ~(zf <= 0.25), Dc, D)
        # unverified columns (only one boundary visible) could be a doorway: a floor-only column may
        # look through a door, so it is trusted only below door-head height (2.0 m); a ceiling-only
        # column only above it (2.12 m). The segmentation band (>= 2.10 m) thus never gets
        # through-door points.
        zD = hf + zr * np.nan_to_num(D)
        D = np.where(only_f & (zD > 2.0), np.nan, D)
        D = np.where(only_c & (zD < 2.12), np.nan, D)
        wall_rng = np.where(wall & np.isfinite(D), D, 0.0)
        # wall points must lie between floor and ceiling
        zw = zr * wall_rng
        wall_rng[(zw < -hf - 0.02) | (zw > hc + 0.02)] = 0.0
        rng = np.where(wall, wall_rng, rng)
        rng[~(fl | ce | wall)] = 0.0
        # convert horizontal range -> depth along -z_cam:  point = t + dw * (rng / nxy)
        s = rng / nxy
        # camera-frame z of the ray (= -1 by construction)
        dcz = (self.dw @ R)[..., 2]
        depth = -dcz * s
        depth[~np.isfinite(depth)] = 0.0
        return depth.astype(np.float32)


def estimate_floor_height(layouts: List[FrameLayout], poses: List[np.ndarray],
                          lo: float = 0.9, hi: float = 2.0, window: int = 24) -> Tuple[float, Dict]:
    """VIO tier: pick the camera height above the floor that makes floor-wall boundary points from
    different camera positions coincide (sharpest Manhattan-aligned histograms in local windows)."""
    cands = np.arange(lo, hi + 1e-9, 0.01)
    score = np.zeros(len(cands))
    used = 0
    for s in range(0, len(layouts), window // 2):
        idx = list(range(s, min(len(layouts), s + window)))
        cams = np.array([poses[i][:2, 3] for i in idx])
        if np.linalg.norm(cams.max(0) - cams.min(0)) < 0.5:
            continue   # spinning in place carries no scale information
        rel = []
        for i in idx:
            L = layouts[i]
            if L.f_ok.sum() == 0:
                continue
            a, r1 = L.f_az[L.f_ok], L.f_r1[L.f_ok]
            rel.append((poses[i][:2, 3], np.cos(a) * r1, np.sin(a) * r1))
        if len(rel) < 4:
            continue
        ref = np.vstack(
            [np.column_stack([c[0] + 1.4 * x, c[1] + 1.4 * y]) for c, x, y in rel])
        ang = dominant_angle(ref)
        if ang is None:
            continue
        ca, sa = np.cos(-ang), np.sin(-ang)
        for j, h in enumerate(cands):
            P = np.vstack([np.column_stack([c[0] + h * x, c[1] + h * y])
                          for c, x, y in rel])
            u = P[:, 0] * ca - P[:, 1] * sa
            v = P[:, 0] * sa + P[:, 1] * ca
            hu = np.bincount(
                ((u - u.min()) / 0.02).astype(np.int64)).astype(float)
            hv = np.bincount(
                ((v - v.min()) / 0.02).astype(np.int64)).astype(float)
            score[j] += ((hu ** 2).sum() + (hv ** 2).sum()) / len(P) ** 2
        used += 1
    if used == 0:
        return 1.40, {"method": "prior (no translating window)", "windows": 0}
    j = int(np.argmax(score))
    return float(cands[j]), {"method": "multi-view floor-boundary consistency", "windows": used,
                             "score_peak_ratio": round(float(score[j] / max(np.median(score), 1e-12)), 3)}
