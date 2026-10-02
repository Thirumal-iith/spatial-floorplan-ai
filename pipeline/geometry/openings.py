"""
pipeline/geometry/openings.py
High-precision opening detector for doors and windows.
Extracts jamb and header coordinates from wall point clouds and 2D projections.
Designed to meet the gate: Opening widths <= 2 cm on >= 85% of openings.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class OpeningDetector:
    def __init__(self, bin_resolution_m: float = 0.005):
        """
        bin_resolution_m: 5mm spatial bins along wall horizontal axis for sub-centimeter edge precision.
        """
        self.bin_resolution = bin_resolution_m

    def detect_openings_on_wall(
        self,
        wall_points_uv: np.ndarray,
        wall_length: float,
        wall_height: float,
        min_door_width: float = 0.65,
        max_door_width: float = 1.30,
        min_door_height: float = 1.90,
        refine_jamb_gradient: bool = True
    ) -> List[Dict]:
        """
        wall_points_uv: (N, 2) where u is along wall length [0, wall_length]
                        and v is height above floor [0, wall_height].
        """
        if len(wall_points_uv) < 100:
            return []

        # Discretize along u axis in 5mm bins
        num_bins = int(np.ceil(wall_length / self.bin_resolution))
        u_bins = np.linspace(0, wall_length, num_bins + 1)
        bin_centers = (u_bins[:-1] + u_bins[1:]) / 2.0

        # Filter for points at typical door opening heights (e.g. 0.3m to 1.8m)
        door_band_mask = (wall_points_uv[:, 1] >= 0.3) & (wall_points_uv[:, 1] <= 1.8)
        band_points = wall_points_uv[door_band_mask]

        if len(band_points) == 0:
            return []

        counts, _ = np.histogram(band_points[:, 0], bins=u_bins)
        
        # Smoothed density to detect sustained negative space (void)
        kernel_size = 5
        smoothed = np.convolve(counts, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Typical wall point count in populated areas
        expected_density = np.percentile(smoothed[smoothed > 0], 75) if np.any(smoothed > 0) else 10.0
        void_threshold = expected_density * 0.15  # <15% of normal density indicates an opening void

        is_void = smoothed < void_threshold

        # Find contiguous void segments
        openings = []
        in_void = False
        start_idx = 0

        for i, val in enumerate(is_void):
            if val and not in_void:
                in_void = True
                start_idx = i
            elif not val and in_void:
                in_void = False
                end_idx = i
                raw_u_start = bin_centers[start_idx]
                raw_u_end = bin_centers[end_idx]
                raw_width = raw_u_end - raw_u_start

                # Verify if dimensions correspond to standard doorway or window
                if min_door_width <= raw_width <= max_door_width:
                    # Refine jamb positions using steepest gradient of point counts
                    if refine_jamb_gradient:
                        u_start, u_end = self._refine_jambs(
                            counts, bin_centers, start_idx, end_idx, self.bin_resolution
                        )
                        width = u_end - u_start
                    else:
                        u_start = raw_u_start
                        u_end = raw_u_end
                        width = raw_width

                    # Detect height by inspecting points above opening
                    height = self._estimate_opening_height(
                        wall_points_uv, u_start, u_end, wall_height, min_door_height
                    )

                    offset = (u_start + u_end) / 2.0
                    openings.append({
                        "type": "door" if height >= 1.90 else "window",
                        "u_start": float(u_start),
                        "u_end": float(u_end),
                        "width_m": float(width),
                        "height_m": float(height),
                        "offset_m": float(offset)
                    })

        return openings

    def _refine_jambs(
        self, counts: np.ndarray, bin_centers: np.ndarray, start_idx: int, end_idx: int, res: float
    ) -> Tuple[float, float]:
        """
        Uses 1D gradient extremum to locate true structural jamb edges within ±3 bins.
        """
        grad = np.gradient(counts)
        # Left jamb: maximum negative drop into void
        window_left = max(0, start_idx - 6), min(len(counts), start_idx + 6)
        sub_grad_left = grad[window_left[0]:window_left[1]]
        if len(sub_grad_left) > 0:
            refined_start_idx = window_left[0] + np.argmin(sub_grad_left)
            u_start = bin_centers[refined_start_idx]
        else:
            u_start = bin_centers[start_idx]

        # Right jamb: maximum positive rise out of void
        window_right = max(0, end_idx - 6), min(len(counts), end_idx + 6)
        sub_grad_right = grad[window_right[0]:window_right[1]]
        if len(sub_grad_right) > 0:
            refined_end_idx = window_right[0] + np.argmax(sub_grad_right)
            u_end = bin_centers[refined_end_idx]
        else:
            u_end = bin_centers[end_idx]

        return float(u_start), float(u_end)

    def _estimate_opening_height(
        self, wall_points: np.ndarray, u_start: float, u_end: float, wall_height: float, min_door_height: float
    ) -> float:
        """
        Estimates opening header height by finding lowest points directly above the void.
        """
        in_col_mask = (wall_points[:, 0] >= u_start + 0.05) & (wall_points[:, 0] <= u_end - 0.05)
        pts_in_col = wall_points[in_col_mask]
        if len(pts_in_col) > 0:
            # Lintel point cluster
            lintel_candidates = pts_in_col[pts_in_col[:, 1] >= 1.8]
            if len(lintel_candidates) > 0:
                return float(np.percentile(lintel_candidates[:, 1], 10))
        return float(min(2.05, wall_height))
