"""
pipeline/geometry/plane_detection.py
Extracts architectural planes (walls, ceiling, floor) from 3D points using RANSAC
and enforces Manhattan-World orthogonal constraints.
"""

from typing import List, Tuple, Dict, Optional, Any
import numpy as np


class PlaneDetector:
    def __init__(self, distance_threshold: float = 0.02, min_points: int = 200):
        self.distance_threshold = distance_threshold
        self.min_points = min_points

    def fit_plane_ransac(self, points: np.ndarray, max_iterations: int = 500) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """
        Fit a single plane ax + by + cz + d = 0 using RANSAC.
        Returns: (normal (3,), d, inlier_indices)
        """
        if len(points) < 3:
            return None

        best_inliers = None
        best_plane = None
        num_points = len(points)

        for _ in range(max_iterations):
            idx = np.random.choice(num_points, 3, replace=False)
            p1, p2, p3 = points[idx]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm = np.linalg.norm(normal)
            if norm < 1e-6:
                continue
            normal = normal / norm
            d = -np.dot(normal, p1)

            distances = np.abs(np.dot(points, normal) + d)
            inliers = np.where(distances < self.distance_threshold)[0]

            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_plane = (normal, d)
                if len(best_inliers) > 0.8 * num_points:
                    break

        if best_inliers is None or len(best_inliers) < self.min_points:
            return None

        # Refine plane on inliers via PCA / SVD
        inlier_pts = points[best_inliers]
        centroid = np.mean(inlier_pts, axis=0)
        u, s, vh = np.linalg.svd(inlier_pts - centroid)
        refined_normal = vh[2, :]
        if np.dot(refined_normal, best_plane[0]) < 0:
            refined_normal = -refined_normal
        refined_d = -np.dot(refined_normal, centroid)

        return refined_normal, refined_d, best_inliers

    def extract_room_planes(self, points: np.ndarray) -> Dict[str, Any]:
        """
        Extracts floor, ceiling, and vertical wall planes from 3D point cloud.
        """
        remaining_points = points.copy()
        extracted_planes = []

        # Find horizontal planes (floor and ceiling: normal.z ~ ±1.0)
        z_vals = points[:, 2]
        z_min, z_max = np.percentile(z_vals, 2), np.percentile(z_vals, 98)
        ceiling_height = float(z_max - z_min)

        # Detect wall planes (normal.z ~ 0.0)
        walls = []
        for _ in range(12):  # up to 12 dominant planar surfaces per room
            if len(remaining_points) < self.min_points:
                break
            result = self.fit_plane_ransac(remaining_points)
            if result is None:
                break
            normal, d, inliers = result

            # Check if vertical (wall)
            if abs(normal[2]) < 0.25:
                # Orthogonalize normal to Manhattan-World (horizontal plane)
                normal_2d = normal[:2]
                norm_2d = np.linalg.norm(normal_2d)
                if norm_2d > 1e-4:
                    normal_2d = normal_2d / norm_2d
                    # Snap to nearest 90-degree axis if within 10 degrees
                    angle = np.arctan2(normal_2d[1], normal_2d[0])
                    snap_angle = round(angle / (np.pi / 2)) * (np.pi / 2)
                    if abs(angle - snap_angle) < np.radians(12):
                        normal_2d = np.array([np.cos(snap_angle), np.sin(snap_angle)])

                    wall_pts = remaining_points[inliers]
                    walls.append({
                        "normal": np.array([normal_2d[0], normal_2d[1], 0.0]),
                        "d": d,
                        "points": wall_pts
                    })

            # Remove inliers from remaining set
            mask = np.ones(len(remaining_points), dtype=bool)
            mask[inliers] = False
            remaining_points = remaining_points[mask]

        return {
            "floor_z": float(z_min),
            "ceiling_z": float(z_max),
            "ceiling_height": ceiling_height,
            "walls": walls
        }
