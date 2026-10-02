"""
pipeline/models.py
Defines the strict Pydantic data contract for property reconstruction, damage evaluation,
concealed-damage reasoning, scoping, and uncertainty quantification.
"""

from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field


class ConfidenceInterval(BaseModel):
    val: float = Field(..., description="Estimated nominal metric value")
    ci_95: Tuple[float, float] = Field(..., description="Calibrated 95% confidence interval [lower, upper]")
    unit: str = Field(default="m", description="Unit of measurement (m, m2, etc.)")


class Opening(BaseModel):
    opening_id: str
    type: str = Field(..., description="Type of opening: door, window, portal")
    width_m: ConfidenceInterval
    height_m: Optional[ConfidenceInterval] = None
    offset_along_wall_m: float = Field(..., description="Distance from start of wall to opening center")
    connected_room_id: Optional[str] = Field(None, description="Connected room if door/portal")


class DamageRegion(BaseModel):
    damage_id: str
    damage_class: str = Field(..., description="water_stain, drywall_crack, mold, fire_smoke, peeling_paint")
    extent_m2: ConfidenceInterval
    location_on_surface: Dict[str, float] = Field(
        ...,
        description="Bounding coordinates on surface: u_min, u_max (horizontal), v_min, v_max (height above floor)"
    )
    severity: str = Field(default="moderate", description="minor, moderate, severe")


class Wall(BaseModel):
    wall_id: str
    start_point: Tuple[float, float] = Field(..., description="[x, y] in room local/global frame")
    end_point: Tuple[float, float] = Field(..., description="[x, y] in room local/global frame")
    length_m: ConfidenceInterval
    height_m: ConfidenceInterval
    normal: Tuple[float, float, float] = Field(default=(0.0, 0.0, 1.0), description="Unit outward normal vector")
    openings: List[Opening] = Field(default_factory=list)
    damage_regions: List[DamageRegion] = Field(default_factory=list)
    adjacent_wall_id: Optional[str] = Field(None, description="Paired wall in adjacent room if shared partition")


class Room(BaseModel):
    room_id: str
    name: str
    polygon: List[Tuple[float, float]] = Field(..., description="Ordered 2D boundary vertices (x, y)")
    floor_area_m2: ConfidenceInterval
    ceiling_height_m: ConfidenceInterval
    walls: List[Wall] = Field(default_factory=list)
    center: Tuple[float, float] = Field(default=(0.0, 0.0), description="Center coordinates (x, y)")


class ConcealedDamageFlag(BaseModel):
    flag_id: str
    room_id: str
    surface_id: str = Field(..., description="Wall or ceiling identifier")
    rule_fired: str = Field(..., description="Identifier of the deterministic building-code/restoration rule")
    trigger_evidence: str = Field(..., description="Metric condition that triggered the flag")
    recommended_action: str = Field(..., description="Inspection/demolition instruction")
    severity: str = Field(default="high", description="low, medium, high")


class ScopeLineItem(BaseModel):
    item_id: str
    surface_ref: str = Field(..., description="e.g. room_living/wall_01")
    category: str = Field(..., description="WTR, DRY, PNT, ELEC, INS")
    code: str = Field(..., description="Standardized restoration code, e.g. WTR-DRY-CUT")
    description: str
    quantity: float
    unit: str = Field(..., description="LF, SF, EA")
    estimated_unit_cost: float
    total_cost: float


class StitchedPlan(BaseModel):
    property_id: str
    tier: str = Field(..., description="photos, video, or lidar")
    timestamp: str
    drift_correction_applied: bool = Field(default=True)
    loop_closures_detected: int = Field(default=0)
    total_footprint_m2: ConfidenceInterval
    rooms: List[Room] = Field(default_factory=list)
    concealed_damage_flags: List[ConcealedDamageFlag] = Field(default_factory=list)
    scope_line_items: List[ScopeLineItem] = Field(default_factory=list)
    total_estimated_restoration_cost: float = Field(default=0.0)
    device_matrix_reference: Dict[str, Any] = Field(default_factory=dict)
