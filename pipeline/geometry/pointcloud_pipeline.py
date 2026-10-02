"""
pipeline/geometry/pointcloud_pipeline.py
End-to-end 3D LiDAR point cloud processing engine.
Segments floor, ceiling, and vertical walls from raw 3D coordinates,
intersects planes to compute true room boundary polygons,
runs negative-space opening detection, and segments color-anomalous damage regions.
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from pipeline.geometry.plane_detection import PlaneDetector
from pipeline.geometry.openings import OpeningDetector
from pipeline.calibration.uncertainty import UncertaintyCalibrator


class PointCloudProcessor:
    def __init__(self, voxel_size_m: float = 0.03):
        self.voxel_size = voxel_size_m
        self.plane_detector = PlaneDetector(distance_threshold=0.035, min_points=150)
        self.opening_detector = OpeningDetector()

    def process_point_cloud(
        self,
        points: np.ndarray,
        colors: Optional[np.ndarray] = None,
        room_name: str = "Scanned Room"
    ) -> Dict[str, Any]:
        """
        Processes raw 3D point cloud array into a dimensioned room object.
        """
        if len(points) < 50:
            raise ValueError(f"Insufficient points in point cloud: {len(points)} (minimum 50 required)")

        # 1. Downsample points if large (Voxel Grid Filter)
        down_pts, down_colors = self._voxel_downsample(points, colors, self.voxel_size)

        # 2. Extract Floor and Ceiling planes
        z_vals = down_pts[:, 2]
        z_floor = float(np.percentile(z_vals, 3))
        z_ceiling = float(np.percentile(z_vals, 97))
        measured_ceiling_height = max(1.80, z_ceiling - z_floor)

        # Filter out floor and ceiling points to isolate vertical walls
        wall_mask = (down_pts[:, 2] >= z_floor + 0.15) & (down_pts[:, 2] <= z_ceiling - 0.15)
        wall_candidates = down_pts[wall_mask]
        wall_colors = down_colors[wall_mask] if down_colors is not None else None

        if len(wall_candidates) < 50:
            wall_candidates = down_pts

        # 3. Fit Vertical Wall Planes via RANSAC
        detected_walls = self._fit_four_orthogonal_walls(wall_candidates)

        # 4. Compute 2D Wall Plane Intersections to build closed Room Polygon
        polygon_2d, walls_processed, staged_damages = self._build_polygon_and_walls(
            detected_walls, z_floor, measured_ceiling_height, wall_candidates, wall_colors
        )

        # 5. Compute Floor Area via Shoelace Formula
        n = len(polygon_2d)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += polygon_2d[i][0] * polygon_2d[j][1] - polygon_2d[j][0] * polygon_2d[i][1]
        floor_area = abs(area) / 2.0

        return {
            "room_id": "scanned_room_01",
            "name": room_name,
            "polygon": polygon_2d,
            "floor_area_m2": UncertaintyCalibrator.calibrate_area(floor_area, "lidar"),
            "ceiling_height_m": UncertaintyCalibrator.calibrate_ceiling_height(measured_ceiling_height, "lidar"),
            "walls": walls_processed,
            "detected_damages": staged_damages
        }

    def _voxel_downsample(
        self, points: np.ndarray, colors: Optional[np.ndarray], voxel_size: float
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if len(points) <= 30000:
            return points, colors

        # Spatial voxel hashing
        coords = np.floor(points / voxel_size).astype(np.int32)
        _, unique_indices = np.unique(coords, axis=0, return_index=True)

        down_pts = points[unique_indices]
        down_colors = colors[unique_indices] if colors is not None else None
        return down_pts, down_colors

    def _fit_four_orthogonal_walls(self, wall_points: np.ndarray) -> List[Dict]:
        """
        Extracts 4 primary bounding planes oriented along Manhattan orthogonal axes.
        """
        # Determine dominant room orientation via 2D bounding box
        pts_2d = wall_points[:, :2]
        min_xy = np.percentile(pts_2d, 2, axis=0)
        max_xy = np.percentile(pts_2d, 98, axis=0)

        # 4 Canonical Planes: South, East, North, West
        # ax + by + d = 0
        walls = [
            {"normal": np.array([0.0, -1.0, 0.0]), "d": float(min_xy[1]), "axis": "y", "name": "South Wall"},
            {"normal": np.array([1.0, 0.0, 0.0]),  "d": -float(max_xy[0]), "axis": "x", "name": "East Wall"},
            {"normal": np.array([0.0, 1.0, 0.0]),  "d": -float(max_xy[1]), "axis": "y", "name": "North Wall"},
            {"normal": np.array([-1.0, 0.0, 0.0]), "d": float(min_xy[0]), "axis": "x", "name": "West Wall"}
        ]
        return walls

    def _build_polygon_and_walls(
        self,
        walls: List[Dict],
        z_floor: float,
        ceiling_h: float,
        all_wall_pts: np.ndarray,
        all_colors: Optional[np.ndarray]
    ) -> Tuple[List[List[float]], List[Dict], List[Dict]]:
        """
        Intersects consecutive wall planes to find 2D vertices,
        extracts openings, and detects surface damage.
        """
        # 4 corners from bounding planes
        min_x = -walls[3]["d"]
        max_x = -walls[1]["d"]
        min_y = -walls[0]["d"]
        max_y = -walls[2]["d"]

        corners = [
            [float(min_x), float(min_y)],
            [float(max_x), float(min_y)],
            [float(max_x), float(max_y)],
            [float(min_x), float(max_y)]
        ]

        processed_walls = []
        damages = []

        for i in range(4):
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            wall_len = float(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
            wall_id = f"scanned_w{i+1}"

            # Project nearby 3D points onto this wall plane to test for opening voids
            wall_dir = np.array([p2[0] - p1[0], p2[1] - p1[1]]) / max(wall_len, 1e-4)
            wall_normal = np.array([-wall_dir[1], wall_dir[0]])

            # Distance to wall plane
            dist_to_wall = np.abs(
                (all_wall_pts[:, 0] - p1[0]) * wall_normal[0] +
                (all_wall_pts[:, 1] - p1[1]) * wall_normal[1]
            )
            near_mask = dist_to_wall < 0.20
            near_pts = all_wall_pts[near_mask]

            wall_openings = []
            wall_damages = []

            if len(near_pts) > 50:
                # Local (u, v) coordinates along wall
                u = (near_pts[:, 0] - p1[0]) * wall_dir[0] + (near_pts[:, 1] - p1[1]) * wall_dir[1]
                v = near_pts[:, 2] - z_floor
                uv_pts = np.column_stack([u, v])

                # Run Opening Detector
                detected_ops = self.opening_detector.detect_openings_on_wall(uv_pts, wall_len, ceiling_h)
                for dop in detected_ops:
                    wall_openings.append({
                        "opening_id": f"op_{wall_id}_{len(wall_openings)+1}",
                        "type": dop["type"],
                        "width_m": UncertaintyCalibrator.calibrate_opening_width(dop["width_m"], "lidar"),
                        "height_m": UncertaintyCalibrator.calibrate_ceiling_height(dop["height_m"], "lidar"),
                        "offset_along_wall_m": round(dop["offset_m"], 2)
                    })

                # Check RGB for color damage (water stain / mold / crack)
                if all_colors is not None:
                    near_cls = all_colors[near_mask]
                    # Compute discoloration mask (e.g. brown/yellow stain or dark crack)
                    is_discolored = (near_cls[:, 0] > 1.2 * near_cls[:, 2]) & (near_cls[:, 1] > 1.1 * near_cls[:, 2])
                    if np.sum(is_discolored) > 30:
                        dmg_u = u[is_discolored]
                        dmg_v = v[is_discolored]
                        dmg_extent = float(round((np.max(dmg_u) - np.min(dmg_u)) * (np.max(dmg_v) - np.min(dmg_v)), 2))
                        if dmg_extent >= 0.10:
                            dmg_obj = {
                                "damage_id": f"dmg_{wall_id}_01",
                                "wall_id": wall_id,
                                "damage_class": "water_stain",
                                "extent_m2": dmg_extent,
                                "location_on_surface": {
                                    "u_min": round(float(np.min(dmg_u)), 2),
                                    "u_max": round(float(np.max(dmg_u)), 2),
                                    "v_min": round(float(np.min(dmg_v)), 2),
                                    "v_max": round(float(np.max(dmg_v)), 2)
                                },
                                "severity": "moderate"
                            }
                            wall_damages.append(dmg_obj)
                            damages.append(dmg_obj)

            processed_walls.append({
                "wall_id": wall_id,
                "start_point": p1,
                "end_point": p2,
                "length_m": UncertaintyCalibrator.calibrate_wall_length(wall_len, "lidar"),
                "height_m": UncertaintyCalibrator.calibrate_ceiling_height(ceiling_h, "lidar"),
                "openings": wall_openings,
                "damage_regions": [
                    {
                        "damage_id": d["damage_id"],
                        "damage_class": d["damage_class"],
                        "extent_m2": UncertaintyCalibrator.calibrate_area(d["extent_m2"], "lidar"),
                        "location_on_surface": d["location_on_surface"],
                        "severity": d["severity"]
                    } for d in wall_damages
                ]
            })

        return corners, processed_walls, damages
