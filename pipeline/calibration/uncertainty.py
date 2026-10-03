"""
pipeline/calibration/uncertainty.py
Bayesian uncertainty quantification and calibration engine.
Computes honest, calibrated 95% confidence intervals across all three tiers
(Photos, Video, LiDAR) to prevent 'confident garbage' penalties.
"""

from typing import Dict, Tuple, Any


class UncertaintyCalibrator:
    # Calibrated noise models per tier
    NOISE_MODELS = {
        "lidar": {
            "wall_rel": 0.003,      # 0.3% length-dependent drift
            "wall_abs": 0.008,      # 8mm base depth noise
            "height_abs": 0.012,    # 1.2cm ceiling height noise
            "opening_abs": 0.015,   # 1.5cm opening edge uncertainty
            "area_rel": 0.012       # 1.2% area uncertainty
        },
        "video": {
            "wall_rel": 0.025,      # 2.5% visual-inertial scale drift
            "wall_abs": 0.020,      # 2cm base feature matching noise
            "height_abs": 0.035,    # 3.5cm ceiling height uncertainty
            "opening_abs": 0.035,   # 3.5cm opening uncertainty
            "area_rel": 0.045       # 4.5% area uncertainty
        },
        "photos": {
            "wall_rel": 0.075,      # 7.5% vanishing-line perspective uncertainty
            "wall_abs": 0.050,      # 5cm scale-prior uncertainty
            "height_abs": 0.120,    # 12cm unmeasured ceiling prior uncertainty
            "opening_abs": 0.070,   # 7cm opening uncertainty
            "area_rel": 0.110       # 11% area uncertainty
        }
    }

    @classmethod
    def calibrate_wall_length(cls, length_m: float, tier: str = "lidar") -> Dict[str, Any]:
        cfg = cls.NOISE_MODELS.get(tier.lower(), cls.NOISE_MODELS["lidar"])
        # Fixed face-snap noise (two wall faces) and length-proportional drift
        # are independent error sources, so they add. max() under-covered long
        # walls (LiDAR wall CI coverage 87.5% < 95% on the synthetic benchmark).
        delta = cfg["wall_abs"] + cfg["wall_rel"] * length_m
        return {
            "val": round(float(length_m), 3),
            "ci_95": (round(float(length_m - delta), 3), round(float(length_m + delta), 3)),
            "unit": "m"
        }

    @classmethod
    def calibrate_ceiling_height(cls, height_m: float, tier: str = "lidar") -> Dict[str, Any]:
        cfg = cls.NOISE_MODELS.get(tier.lower(), cls.NOISE_MODELS["lidar"])
        delta = cfg["height_abs"]
        return {
            "val": round(float(height_m), 3),
            "ci_95": (round(float(height_m - delta), 3), round(float(height_m + delta), 3)),
            "unit": "m"
        }

    @classmethod
    def calibrate_opening_width(cls, width_m: float, tier: str = "lidar") -> Dict[str, Any]:
        cfg = cls.NOISE_MODELS.get(tier.lower(), cls.NOISE_MODELS["lidar"])
        delta = cfg["opening_abs"]
        return {
            "val": round(float(width_m), 3),
            "ci_95": (round(float(width_m - delta), 3), round(float(width_m + delta), 3)),
            "unit": "m"
        }

    @classmethod
    def calibrate_area(cls, area_m2: float, tier: str = "lidar") -> Dict[str, Any]:
        cfg = cls.NOISE_MODELS.get(tier.lower(), cls.NOISE_MODELS["lidar"])
        delta = max(0.1, cfg["area_rel"] * area_m2)
        return {
            "val": round(float(area_m2), 2),
            "ci_95": (round(float(area_m2 - delta), 2), round(float(area_m2 + delta), 2)),
            "unit": "m2"
        }
