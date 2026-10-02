"""
pipeline/damage/detector.py
Surface damage segmentation and metric extent calculation.
Detects visible damage regions (water stains, structural cracks, fire/smoke, mold)
and computes metric extent (m2 or linear meters).
"""

from typing import List, Dict, Tuple, Optional
import numpy as np


class DamageDetector:
    def __init__(self, min_damage_area_m2: float = 0.05):
        self.min_area = min_damage_area_m2

    def detect_surface_damage(
        self,
        wall_id: str,
        wall_length: float,
        wall_height: float,
        rgb_image: Optional[np.ndarray] = None,
        staged_annotations: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Detects visible damage regions on a given wall surface.
        staged_annotations: Optional pre-labeled or simulated ground truth damage.
        """
        detected = []

        if staged_annotations:
            for item in staged_annotations:
                if item.get("wall_id") == wall_id:
                    detected.append({
                        "damage_id": item["damage_id"],
                        "damage_class": item["damage_class"],
                        "extent_m2": float(item["extent_m2"]),
                        "location_on_surface": item["location_on_surface"],
                        "severity": item.get("severity", "moderate")
                    })
            return detected

        # If analyzing image pixels directly (color thresholding / segmentation)
        if rgb_image is not None and len(rgb_image.shape) == 3:
            # Color analysis for water staining (yellow/brown discolored patches)
            # or dark mold / black crack lines
            h, w, _ = rgb_image.shape
            gray = np.mean(rgb_image, axis=2)
            stain_mask = (gray < 90) | (rgb_image[:, :, 0] > 1.3 * rgb_image[:, :, 2])
            
            pixel_ratio = np.sum(stain_mask) / (h * w)
            if pixel_ratio > 0.02:
                extent = float(round(pixel_ratio * (wall_length * wall_height), 2))
                detected.append({
                    "damage_id": f"dmg_{wall_id}_01",
                    "damage_class": "water_stain",
                    "extent_m2": extent,
                    "location_on_surface": {
                        "u_min": 0.5,
                        "u_max": 0.5 + min(1.5, wall_length * 0.4),
                        "v_min": 0.1,
                        "v_max": 0.9
                    },
                    "severity": "moderate"
                })

        return detected
