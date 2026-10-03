"""
pipeline/geometry/drift_correction.py

2D pose-graph optimisation for multi-room captures.

Why 2D: floor plans live in the ground plane and the gravity axis is observable from
the IMU on every phone, so drift that matters for a floor plan is yaw + x/y.

Two kinds of constraints correct the odometry:
  1. Plane-anchored (Manhattan) yaw constraints. When a room is observed at keyframe k,
     its dominant wall direction in world must be a multiple of 90 deg. The difference
     gives an absolute yaw measurement at k.
  2. Loop closures. The capture protocol starts at a marked spot in the connector and
     returns over it between rooms and at the end. Revisits of the start position
     (detected after yaw correction) become position-equality constraints.

Solved as two linear least-squares problems (yaw first, then translation). This is
the standard linearised SE(2) decoupling and is exact for the yaw subproblem.

apply_drift_correction=False returns the raw poses ("poses as-is") for the ablation.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


def _wrap(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(T: np.ndarray) -> float:
    """Heading proxy used everywhere: direction of the pose's x-axis in the ground plane."""
    return float(np.arctan2(T[1, 0], T[0, 0]))


def se2_of(T: np.ndarray):
    """(2x2 rotation, 2-vector) ground-plane transform of a z-up pose."""
    a = yaw_of(T)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]]), np.asarray(T[:2, 3], dtype=float)


def _to_se2(poses: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Return unwrapped yaw (N,) and xy (N,2)."""
    raw = np.array([np.arctan2(T[1, 0], T[0, 0]) for T in poses])
    theta = np.empty_like(raw)
    theta[0] = raw[0]
    for i in range(1, len(raw)):
        theta[i] = theta[i - 1] + _wrap(raw[i] - raw[i - 1])
    xy = np.array([[T[0, 3], T[1, 3]] for T in poses])
    return theta, xy


def _from_se2(theta: np.ndarray, xy: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
    """Apply the yaw change as a rotation about gravity, keeping pitch/roll and height."""
    theta_old, _ = _to_se2(template)
    out = []
    for th, th0, p, T in zip(theta, theta_old, xy, template):
        M = np.array(T, dtype=float).copy()
        d = th - th0
        c, s = np.cos(d), np.sin(d)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        M[:3, :3] = Rz @ M[:3, :3]
        M[0, 3], M[1, 3] = p
        out.append(M)
    return out


class PoseGraphOptimizer:
    def __init__(self, loop_radius_m: float = 0.5, min_loop_gap: int = 20,
                 w_odom: float = 1.0, w_anchor: float = 30.0, w_loop: float = 30.0):
        self.loop_radius = loop_radius_m
        self.min_gap = min_loop_gap
        self.w_odom, self.w_anchor, self.w_loop = w_odom, w_anchor, w_loop

    # ------------------------------------------------------------------ public
    def optimize_trajectory(
        self,
        raw_poses: List[np.ndarray],
        detected_wall_planes: Optional[List[Dict]] = None,
        apply_drift_correction: bool = True,
        manhattan_anchors: Optional[List[Tuple[int, float]]] = None,
        loop_closure_hints: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[List[np.ndarray], Dict]:
        """
        manhattan_anchors: (keyframe_index, wall_direction_in_camera_frame_rad)
        loop_closure_hints: optional explicit (i, j) position-equality pairs
        """
        raw_poses = [np.asarray(p, dtype=float) for p in raw_poses]
        n = len(raw_poses)
        closure_err_raw = self._closure_error(raw_poses)
        if not apply_drift_correction or n < 3:
            return raw_poses, {
                "drift_correction_applied": False,
                "method": "raw_poses_as_is",
                "loop_closures_found": 0,
                "manhattan_anchors_used": 0,
                "residual_drift_m": closure_err_raw,
            }

        theta, xy = _to_se2(raw_poses)
        dtheta = np.diff(theta)
        # odometry translation expressed in the frame of pose i
        dt_local = np.array(
            [self._rot(-theta[i]) @ (xy[i + 1] - xy[i]) for i in range(n - 1)])

        # ---- 1. yaw: odometry + Manhattan anchors. The Manhattan frame is defined by the
        # earliest anchor (least drift), so no assumption about the world axes is needed.
        anchors = []
        quarter = np.pi / 2
        ref = None
        srt = sorted(a for a in (manhattan_anchors or []) if 0 <= a[0] < n)
        if srt:
            # Manhattan reference = the earliest anchor (least drift; keeps the start heading as the
            # world gauge), unless it disagrees with the next two by > 1.5 deg - then it is a
            # mis-detection and the mod-90 median of the first three is used.
            first = np.array([theta[k] + la for k, la in srt[:3]])
            def wrapq(x): return (x + np.pi / 4) % (np.pi / 2) - np.pi / 4
            ref = float(first[0])
            if len(first) == 3 and np.all(np.abs(wrapq(first[1:] - first[0])) > np.radians(1.5)):
                cost = [np.abs(wrapq(first - f)).sum() for f in first]
                ref = float(first[int(np.argmin(cost))])
        for k, local_angle in srt:
            a = theta[k] + local_angle
            snapped = ref + np.round((a - ref) / quarter) * quarter
            anchors.append((k, theta[k] + (snapped - a)))
        # Robustness: the yaw correction (target - raw yaw) changes slowly because drift accumulates
        # gradually. An anchor whose correction disagrees with its neighbours' median by > 1.5 deg
        # is a mis-detected wall direction and is dropped before solving.
        n_rejected = 0
        if len(anchors) >= 5:
            corr = np.array([tgt - theta[k] for k, tgt in anchors])
            keep = np.ones(len(anchors), bool)
            for i in range(len(anchors)):
                nb = np.r_[corr[max(0, i - 5):i], corr[i + 1:i + 6]]
                if len(nb) >= 3 and abs(corr[i] - np.median(nb)) > np.radians(1.5):
                    keep[i] = False
            n_rejected = int((~keep).sum())
            anchors = [a for a, ok in zip(anchors, keep) if ok]
        theta_c = self._solve_yaw(theta, dtheta, anchors)

        # ---- 2. re-integrate with corrected yaw, detect loop closures
        xy_pred = self._integrate(xy[0], theta_c, dt_local)
        closures = list(loop_closure_hints or []
                        ) or self._detect_closures(xy_pred)

        # ---- 3. translation: odometry + closures
        xy_c = self._solve_xy(xy[0], theta_c, dt_local, closures)
        corrected = _from_se2(theta_c, xy_c, raw_poses)

        return corrected, {
            "drift_correction_applied": True,
            "method": "se2_pose_graph_manhattan_yaw_anchors_plus_loop_closure",
            "loop_closures_found": len(closures),
            "loop_closure_pairs": [list(map(int, c)) for c in closures],
            "manhattan_anchors_used": len(anchors),
            "manhattan_anchors_rejected": n_rejected,
            "max_yaw_correction_deg": float(np.degrees(np.max(np.abs(theta_c - theta)))) if n else 0.0,
            "initial_drift_m": closure_err_raw,
            "residual_drift_m": self._closure_error(corrected),
        }

    # ----------------------------------------------------------------- solvers
    def _solve_yaw(self, theta, dtheta, anchors):
        n = len(theta)
        rows, rhs = [], []
        for i, d in enumerate(dtheta):
            r = np.zeros(n)
            r[i + 1], r[i] = self.w_odom, -self.w_odom
            rows.append(r)
            rhs.append(self.w_odom * d)
        gauge_w = 1e-3 if anchors else 1e3
        r = np.zeros(n)
        r[0] = gauge_w
        rows.append(r)
        rhs.append(gauge_w * theta[0])
        for k, target in anchors:
            r = np.zeros(n)
            r[k] = self.w_anchor
            rows.append(r)
            rhs.append(self.w_anchor * target)
        return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

    def _solve_xy(self, p0, theta, dt_local, closures):
        n = len(theta)
        rows, rhs = [], []
        for i in range(n - 1):
            d = self._rot(theta[i]) @ dt_local[i]
            r = np.zeros(n)
            r[i + 1], r[i] = self.w_odom, -self.w_odom
            rows.append(r)
            rhs.append(self.w_odom * d)
        r = np.zeros(n)
        r[0] = 1e3
        rows.append(r)
        rhs.append(1e3 * np.asarray(p0))
        for i, j in closures:
            r = np.zeros(n)
            r[j], r[i] = self.w_loop, -self.w_loop
            rows.append(r)
            rhs.append(np.zeros(2))
        return np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

    def _detect_closures(self, xy):
        """Revisits of the start marker: local minima of distance to pose 0 within radius."""
        d = np.linalg.norm(xy - xy[0], axis=1)
        closures, last = [], 0
        for j in range(self.min_gap, len(d)):
            lo, hi = max(0, j - 10), min(len(d), j + 11)
            if d[j] < self.loop_radius and d[j] == d[lo:hi].min() and j - last >= self.min_gap:
                closures.append((0, j))
                last = j
        return closures

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _rot(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s], [s, c]])

    def _integrate(self, p0, theta, dt_local):
        out = [np.asarray(p0, dtype=float)]
        for i, d in enumerate(dt_local):
            out.append(out[-1] + self._rot(theta[i]) @ d)
        return np.array(out)

    @staticmethod
    def _closure_error(poses) -> float:
        if len(poses) < 2:
            return 0.0
        return float(np.linalg.norm(poses[-1][:2, 3] - poses[0][:2, 3]))
