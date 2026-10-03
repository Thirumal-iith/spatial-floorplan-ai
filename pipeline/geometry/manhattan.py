"""
pipeline/geometry/manhattan.py
Dominant (Manhattan) wall orientation of a set of ground-plane points.

Wall points projected on the floor form lines. When the axes are aligned with the walls, the
1D histograms of x and y are maximally peaked; we maximise sum(hist^2) over angles in [0, 90).
"""

from typing import Optional
import numpy as np


def _score(xy: np.ndarray, a: float, res: float) -> float:
    c, s = np.cos(a), np.sin(a)
    u = xy[:, 0] * c + xy[:, 1] * s
    v = -xy[:, 0] * s + xy[:, 1] * c
    hu = np.bincount(((u - u.min()) / res).astype(np.int64)).astype(np.float64)
    hv = np.bincount(((v - v.min()) / res).astype(np.int64)).astype(np.float64)
    return float((hu ** 2).sum() + (hv ** 2).sum())


def dominant_angle(xy: np.ndarray, res: float = 0.02, max_pts: int = 30000, seed: int = 0) -> Optional[float]:
    """Return wall direction in radians in [0, pi/2), or None if too few points."""
    xy = np.asarray(xy, dtype=np.float64)
    if len(xy) < 50:
        return None
    if len(xy) > max_pts:
        xy = xy[np.random.default_rng(seed).choice(len(xy), max_pts, replace=False)]
    xy = xy - xy.mean(axis=0)
    coarse = np.radians(np.arange(0.0, 90.0, 0.5))
    a0 = coarse[int(np.argmax([_score(xy, a, res) for a in coarse]))]
    fine = a0 + np.radians(np.arange(-0.5, 0.5001, 0.02))
    best = fine[int(np.argmax([_score(xy, a, res) for a in fine]))]
    return float(best % (np.pi / 2))
