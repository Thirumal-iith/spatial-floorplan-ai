"""
pipeline/geometry/lidar_layout.py

LiDAR-tier perception: raw depth frames + (drifting) poses -> dimensioned, stitched multi-room plan.

  1. Floor height and lowest ceiling from the height histogram (gravity is +z after ingestion).
  2. Drift: per-keyframe Manhattan wall direction -> yaw anchors; hub revisits -> loop closures;
     solved by PoseGraphOptimizer. With drift correction OFF the raw poses are used as-is.
  3. Fuse all frames into one world cloud with the chosen poses.
  4. Room segmentation in plan view: observed floor/ceiling cells minus a wall mask built from a
     height band ABOVE door heads and BELOW the lowest ceiling. Door heads close every doorway in
     that band, so each room becomes its own connected free-space region (no learned model needed).
  5. Per room: rectilinear polygon from the region contour, every edge then snapped to the wall face
     (5 mm histogram peak of points seen from inside the room).
  6. Per room: floor and ceiling planes from height-histogram peaks inside the region.
  7. Per wall: openings = rectangular voids in the wall-face occupancy image; jambs/head/sill
     measured from the raw points; doors linked to the room on the other side (adjacency).
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import cv2

from pipeline.ingestion.lidar_capture import load_capture, backproject
from pipeline.geometry.manhattan import dominant_angle
from pipeline.geometry.drift_correction import PoseGraphOptimizer, _to_se2
from pipeline.calibration.uncertainty import UncertaintyCalibrator as UC


def _rot(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _peak_refine(vals: np.ndarray, res: float, win: float) -> Optional[float]:
    if len(vals) < 20:
        return None
    lo, hi = float(vals.min()), float(vals.max())
    bins = np.arange(lo, hi + res, res)
    if len(bins) < 2:
        return float(np.median(vals))
    h, e = np.histogram(vals, bins=bins)
    c = (e[int(np.argmax(h))] + e[int(np.argmax(h)) + 1]) / 2
    sel = vals[np.abs(vals - c) < win]
    return float(np.median(sel)) if len(sel) else float(c)


class LidarReconstructor:
    GRID = 0.02
    MIN_ROOM_M2 = 1.5
    WALL_DILATE_PX = 4      # 8 cm: closes wall cavities up to ~16 cm
    OBS_CLOSE_PX = 11       # 22 cm: fills floor/ceiling coverage gaps
    FACE_TOL = 0.03
    OPEN_GRID = 0.03

    def __init__(self, tier: str = "lidar", align_output: bool = False, scan_refine: bool = False):
        self.pgo = PoseGraphOptimizer()
        self.tier = tier
        # True: keep the Manhattan-aligned frame (photo tier)
        self.align_output = align_output
        # True: scan-to-map xy refinement after the pose graph
        self.scan_refine = scan_refine
        if tier != "lidar":
            # pseudo depth from mono layout is noisier than LiDAR: wider face slab, coarser grid
            self.FACE_TOL = 0.08
            self.OPEN_GRID = 0.05

    # ------------------------------------------------------------- scan-to-map refinement
    def _scan_refine(self, frames, poses, floor_z, ceil_min, search_m: float = 0.12):
        """Sequential 2D scan-to-map alignment of wall points (classic laser-scan matching on a grid).
        Corrects the slowly varying translation error (VIO scale/yaw residue) the hub loop closures
        cannot see inside rooms. Each frame inherits the previous offset and searches +-12 cm."""
        g = 0.02
        scans = []
        for f, P in zip(frames, poses):
            p = backproject(f, P, stride=3)
            p = p[(p[:, 2] > floor_z + 0.05) &
                  (p[:, 2] < ceil_min - 0.05)][:, :2]
            scans.append(p)
        allp = np.vstack([s for s in scans if len(s)])
        mn = allp.min(axis=0) - 1.5
        W, H = (np.ceil((allp.max(axis=0) + 1.5 - mn) / g)).astype(int) + 1
        grid = np.zeros((H, W), np.float32)
        r = int(round(search_m / g))
        off = np.zeros(2)
        out, shifts, matched = [], [], 0
        for s, P in zip(scans, poses):
            Q = P.copy()
            Q[:2, 3] += off
            if len(s) >= 150 and grid.sum() > 3000:
                q = np.floor((s + off - mn) / g).astype(int)
                x0, y0 = q.min(axis=0)
                x1, y1 = q.max(axis=0) + 1
                tmpl = np.zeros((y1 - y0, x1 - x0), np.float32)
                np.add.at(tmpl, (q[:, 1] - y0, q[:, 0] - x0), 1.0)
                tmpl = cv2.GaussianBlur(tmpl, (5, 5), 1.0)
                ya, yb, xa, xb = y0 - r, y1 + r, x0 - r, x1 + r
                if ya >= 0 and xa >= 0 and yb <= H and xb <= W:
                    crop = cv2.GaussianBlur(np.minimum(
                        grid[ya:yb, xa:xb], 5.0), (5, 5), 1.0)
                    res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCORR)
                    j, i = np.unravel_index(int(np.argmax(res)), res.shape)
                    centre = res[r, r]
                    if res[j, i] > 1.05 * centre and res[j, i] > 0 and (tmpl * crop[r:r + tmpl.shape[0], r:r + tmpl.shape[1]] > 0).mean() > 0.002:
                        d = np.array([(i - r) * g, (j - r) * g])
                        off = off + d
                        Q[:2, 3] += d
                        matched += 1
            q = np.floor((s + (Q[:2, 3] - P[:2, 3]) - mn) / g).astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (
                q[:, 1] >= 0) & (q[:, 1] < H)
            np.add.at(grid, (q[ok, 1], q[ok, 0]), 1.0)
            shifts.append(float(np.linalg.norm(Q[:2, 3] - P[:2, 3])))
            out.append(Q)
        return out, {"method": "sequential 2D scan-to-map grid correlation (wall points)",
                     "frames_shifted": matched, "max_total_shift_m": round(max(shifts), 3)}

    # ------------------------------------------------------------------ main
    def reconstruct(self, path: str, drift_correction: bool = True) -> Tuple[List[Dict], Dict, Dict]:
        return self.reconstruct_frames(load_capture(path), drift_correction)

    def reconstruct_frames(self, frames: List[Dict], drift_correction: bool = True,
                           anchors: Optional[List[Tuple[int, float]]] = None) -> Tuple[List[Dict], Dict, Dict]:
        if len(frames) < 1:
            raise ValueError("capture has no usable frames")
        raw = [f["pose"] for f in frames]

        sample = np.vstack([backproject(f, stride=8) for f in frames])
        floor_z = self._floor(sample[:, 2])
        ceil_min = self._ceiling_min(sample[:, 2], floor_z)

        # --- drift: Manhattan yaw anchors from individual keyframes (raw poses)
        theta, _ = _to_se2(raw)
        if anchors is None:
            anchors = []
            for k in range(0, len(frames), 2):
                p = backproject(frames[k], stride=4)
                p = p[(p[:, 2] > floor_z + 0.3) & (p[:, 2] < ceil_min - 0.15)]
                if len(p) >= 300:
                    a = dominant_angle(p[:, :2])
                    if a is not None:
                        anchors.append((k, a - theta[k]))
        poses, drift_meta = self.pgo.optimize_trajectory(
            raw, apply_drift_correction=drift_correction, manhattan_anchors=anchors)
        if drift_correction and self.scan_refine:
            poses, refine_meta = self._scan_refine(
                frames, poses, floor_z, ceil_min)
            drift_meta["scan_to_map_refinement"] = refine_meta

        # --- fused cloud
        pts_l, cam_l = [], []
        for f, P in zip(frames, poses):
            p = backproject(f, P, stride=2)
            pts_l.append(p)
            cam_l.append(np.broadcast_to(
                P[:2, 3].astype(np.float32), (len(p), 2)))
        pts = np.vstack(pts_l)
        cams = np.vstack(cam_l)
        z = pts[:, 2]

        band = (floor_z + 2.10, ceil_min - 0.06)
        band_ok = band[1] - band[0] >= 0.08
        if not band_ok:
            band = (floor_z + 1.0, floor_z + 1.9)
        band_m = (z > band[0]) & (z < band[1])
        ang = dominant_angle(pts[band_m, :2]) or 0.0

        R = _rot(-ang)
        xy = pts[:, :2] @ R.T
        cam = cams @ R.T

        rooms_px, labels, mn = self._segment(xy, z, band_m, floor_z, ceil_min)
        lab = self._lookup(labels, mn, xy)
        # which room each observing camera stood in: walls/openings of a room are measured only from
        # frames taken inside it (short time window -> least relative drift, no views through doors)
        cam_lab = self._lookup(labels, mn, cam)

        rooms, polys = [], {}
        wall_z = (z > floor_z + 0.2) & (z < ceil_min - 0.1)
        for idx, (lbl, area_px) in enumerate(rooms_px):
            inside = cam_lab == lbl
            if inside.sum() < 500:
                inside = np.ones(len(cam), bool)
            V = self._room_polygon(labels == lbl, mn, xy, cam, wall_z & inside)
            if V is None:
                continue
            rid = f"room_{idx + 1:02d}"
            zr = z[lab == lbl]
            f_r = _peak_refine(zr[zr < floor_z + 0.15],
                               0.005, 0.015) or floor_z
            c_r = _peak_refine(zr[zr > floor_z + 1.9],
                               0.005, 0.015) or ceil_min
            rooms.append({"rid": rid, "lbl": lbl, "V": V, "floor": f_r,
                         "ceil": c_r, "inside": (xy[inside], z[inside], cam[inside])})
            polys[lbl] = rid

        out = []
        Rb = np.eye(2) if self.align_output else _rot(ang)
        for r in rooms:
            V, f_r, c_r = r["V"], r["floor"], r["ceil"]
            h = c_r - f_r
            walls = []
            n = len(V)
            for i in range(n):
                a, b = V[i], V[(i + 1) % n]
                L = float(np.linalg.norm(b - a))
                if L < 0.05:
                    continue
                wid = f"{r['rid']}_w{len(walls) + 1}"
                sx, sz, sc = r["inside"]
                ops = self._openings(a, b, V, sx, sz, sc,
                                     f_r, c_r, labels, mn, polys, wid)
                aw, bw = Rb @ a, Rb @ b
                walls.append({
                    "wall_id": wid,
                    "start_point": [round(float(aw[0]), 4), round(float(aw[1]), 4)],
                    "end_point": [round(float(bw[0]), 4), round(float(bw[1]), 4)],
                    "length_m": UC.calibrate_wall_length(L, self.tier),
                    "height_m": UC.calibrate_ceiling_height(h, self.tier),
                    "openings": ops,
                    "damage_regions": [],
                })
            Vw = (V @ Rb.T).round(4).tolist()
            ext = V.max(axis=0) - V.min(axis=0)
            hallway = min(ext) < 1.8 or max(ext) / max(min(ext), 1e-6) > 2.2
            out.append({
                "room_id": r["rid"],
                "name": ("Hallway " if hallway else "Room ") + r["rid"][-2:],
                "room_type": "hallway" if hallway else "room",
                "polygon": Vw,
                "ceiling_height_m": round(h, 4),
                "walls": walls,
                "pose_placed": True,
                "detected_damages": [],
            })

        telemetry = {
            "frames": len(frames), "points": int(len(pts)),
            "floor_z_m": round(floor_z, 4), "lowest_ceiling_z_m": round(ceil_min, 4),
            "segmentation_band_m": [round(band[0] - floor_z, 3), round(band[1] - floor_z, 3)],
            "segmentation_band_above_doors": band_ok,
            "manhattan_angle_deg": round(float(np.degrees(ang)), 3),
            "manhattan_anchor_frames": len(anchors),
        }
        return out, drift_meta, telemetry

    # ------------------------------------------------------------ heights
    @staticmethod
    def _floor(z: np.ndarray) -> float:
        lo = z[z < np.median(z)]
        return _peak_refine(lo, 0.01, 0.02) or float(np.percentile(z, 1))

    @staticmethod
    def _ceiling_min(z: np.ndarray, floor_z: float) -> float:
        bins = np.arange(floor_z, floor_z + 4.0, 0.01)
        h, e = np.histogram(z, bins=bins)
        centers = (e[:-1] + e[1:]) / 2
        ref = np.median(h[(centers > floor_z + 0.5) &
                        (centers < floor_z + 1.8)]) + 1
        cand = np.where((centers > floor_z + 1.9) & (h > 4 * ref))[0]
        if len(cand) == 0:
            return float(np.percentile(z, 99))
        return float(centers[cand[0]])

    # ------------------------------------------------------------ segmentation
    def _segment(self, xy, z, band_m, floor_z, ceil_min):
        g = self.GRID
        mn = xy.min(axis=0) - 0.3
        size = (np.ceil((xy.max(axis=0) + 0.3 - mn) / g)).astype(int) + 1
        W, H = int(size[0]), int(size[1])

        def raster(mask, min_count):
            q = np.floor((xy[mask] - mn) / g).astype(np.int64)
            cnt = np.bincount(q[:, 1] * W + q[:, 0],
                              minlength=W * H).reshape(H, W)
            return (cnt >= min_count).astype(np.uint8)

        wall = raster(band_m, 2)
        obs = raster((np.abs(z - floor_z) < 0.04) | (z > ceil_min - 0.04), 1)
        def ell(r): return cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        obs = cv2.morphologyEx(obs, cv2.MORPH_CLOSE,
                               ell(self.OBS_CLOSE_PX // 2))
        wall_d = cv2.dilate(wall, ell(self.WALL_DILATE_PX))
        free = (obs & (1 - wall_d)).astype(np.uint8)
        free = cv2.morphologyEx(free, cv2.MORPH_OPEN, ell(1))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            free, connectivity=4)
        min_px = self.MIN_ROOM_M2 / g ** 2
        rooms = [(i, int(stats[i, cv2.CC_STAT_AREA]))
                 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_px]
        rooms.sort(key=lambda t: -t[1])
        return rooms, labels, mn

    def _lookup(self, labels, mn, xy):
        q = np.floor((xy - mn) / self.GRID).astype(np.int64)
        H, W = labels.shape
        ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        out = np.zeros(len(xy), dtype=labels.dtype)
        out[ok] = labels[q[ok, 1], q[ok, 0]]
        return out

    # ------------------------------------------------------------ polygon
    def _room_polygon(self, mask, mn, xy, cam, wall_z) -> Optional[np.ndarray]:
        g = self.GRID
        cnts, _ = cv2.findContours(mask.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        approx = cv2.approxPolyDP(
            c, (0.10 if self.tier == "lidar" else 0.22) / g, True).reshape(-1, 2).astype(float)
        P = (approx + 0.5) * g + mn
        edges = self._rectilinear(P)
        if len(edges) < 4 or len(edges) % 2:
            x, y, w, h = cv2.boundingRect(c)
            x0, y0 = mn + np.array([x, y]) * g
            x1, y1 = mn + np.array([x + w, y + h]) * g
            edges = [("h", y0), ("v", x1), ("h", y1), ("v", x0)]
        V = self._vertices(edges)
        edges = [self._snap_edge(e, V[i], V[(i + 1) % len(V)], V, xy, cam, wall_z)
                 for i, e in enumerate(edges)]
        return self._vertices(edges)

    @staticmethod
    def _rectilinear(P):
        edges = []
        n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]
            d = b - a
            L = float(np.hypot(*d))
            if L < 1e-6:
                continue
            edges.append(["h", (a[1] + b[1]) / 2, L] if abs(d[0])
                         >= abs(d[1]) else ["v", (a[0] + b[0]) / 2, L])
        changed = True
        while changed and len(edges) > 1:
            changed = False
            for i in range(len(edges)):
                j = (i + 1) % len(edges)
                if i != j and edges[i][0] == edges[j][0]:
                    w = edges[i][2] + edges[j][2]
                    edges[i] = [
                        edges[i][0], (edges[i][1] * edges[i][2] + edges[j][1] * edges[j][2]) / w, w]
                    del edges[j]
                    changed = True
                    break
        return [(o, c) for o, c, _ in edges]

    @staticmethod
    def _vertices(edges) -> np.ndarray:
        V = []
        for i, (o, c) in enumerate(edges):
            po, pc = edges[i - 1]
            V.append([c, pc] if (po == "h" and o == "v") else [pc, c])
        return np.array(V, dtype=float)

    @staticmethod
    def _outward(a, b, V):
        d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        n = np.array([-d[1], d[0]])
        mid = (a + b) / 2 + 0.05 * n
        if cv2.pointPolygonTest(V.astype(np.float32).reshape(-1, 1, 2), (float(mid[0]), float(mid[1])), False) > 0:
            n = -n
        return d, n

    def _snap_edge(self, e, a, b, V, xy, cam, wall_z):
        o, c = e
        L = float(np.linalg.norm(b - a))
        if L < 0.4:
            return e
        d, n = self._outward(a, b, V)
        rel = xy - a
        along = rel @ d
        s = rel @ n
        m = (along > 0.15) & (along < L - 0.15) & (s > -0.15) & (s < 0.35) & wall_z \
            & (((cam - a) @ n) < -0.05)
        if m.sum() < 50:
            return e
        off = _peak_refine(s[m], 0.005, 0.015)
        if off is None:
            return e
        return (o, c + off * (n[1] if o == "h" else n[0]))

    # ------------------------------------------------------------ openings
    def _openings(self, a, b, V, xy, z, cam, f_r, c_r, labels, mn, polys, wid) -> List[Dict]:
        g = self.OPEN_GRID
        L = float(np.linalg.norm(b - a))
        h = c_r - f_r
        if L < 0.6:
            return []
        d, n = self._outward(a, b, V)
        rel = xy - a
        s = rel @ n
        m = (np.abs(s) < self.FACE_TOL) & (
            z > f_r + 0.04) & (z < c_r - 0.04) & (((cam - a) @ n) < -0.05)
        u = (rel[m] @ d)
        v = z[m] - f_r
        k = (u >= 0) & (u <= L)
        u, v = u[k], v[k]
        if len(u) < 200:
            return []
        nu, nv = int(np.ceil(L / g)), int(np.ceil(h / g))
        qi = np.clip((u / g).astype(int), 0, nu - 1)
        qj = np.clip((v / g).astype(int), 0, nv - 1)
        occ = np.zeros((nv, nu), dtype=np.uint8)
        occ[qj, qi] = 1
        k_close = 3 if self.tier == "lidar" else 5
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE,
                               np.ones((k_close, k_close), np.uint8))
        empty = (1 - occ).astype(np.uint8)
        cnt, _, stats, _ = cv2.connectedComponentsWithStats(
            empty, connectivity=4)
        ops = []
        for i in range(1, cnt):
            x, y, w, hh, area = stats[i]
            if w * g < 0.45 or hh * g < 0.45 or area / float(w * hh) < 0.75:
                continue
            if x == 0 or x + w >= nu:
                continue
            u0, u1, v0, v1 = x * g, (x + w) * g, y * g, (y + hh) * g
            uc, vc = (u0 + u1) / 2, (v0 + v1) / 2
            vb = (v > v0 + 0.1) & (v < v1 - 0.1)
            lc = u[vb & (u < uc) & (u > u0 - 0.25)]
            rc = u[vb & (u > uc) & (u < u1 + 0.25)]
            if len(lc) < 10 or len(rc) < 10:
                continue
            left, right = float(np.percentile(lc, 99)), float(
                np.percentile(rc, 1))
            col = (u > left + 0.05) & (u < right - 0.05)
            tc, bc = v[col & (v > vc)], v[col & (v < vc)]
            top = float(np.percentile(tc, 1)) if len(tc) >= 10 else v1
            bottom = float(np.percentile(bc, 99)) if len(bc) >= 10 else 0.0
            if bottom < 0.10:
                bottom = 0.0
            # Plausibility: a void reaching the floor must be door-height; a window needs a sill.
            # Low floor-level voids are almost always unobserved wall behind furniture -> rejected.
            if bottom == 0.0 and top < 1.8:
                continue
            if 0.0 < bottom < 0.3:
                continue
            kind = "door" if bottom == 0.0 else "window"
            probe = a + d * ((left + right) / 2) + n * 0.4
            q = np.floor((probe - mn) / self.GRID).astype(int)
            other = None
            if 0 <= q[1] < labels.shape[0] and 0 <= q[0] < labels.shape[1]:
                other = polys.get(int(labels[q[1], q[0]]))
            ops.append({
                "opening_id": f"{wid}_op{len(ops) + 1}",
                "type": kind,
                "width_m": UC.calibrate_opening_width(right - left, self.tier),
                "height_m": UC.calibrate_ceiling_height(top - bottom, self.tier),
                "sill_m": round(bottom, 3),
                "offset_along_wall_m": round((left + right) / 2, 3),
                "connected_room_id": other if kind == "door" else None,
            })
        return ops
