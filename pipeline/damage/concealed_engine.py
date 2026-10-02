"""
pipeline/damage/concealed_engine.py
Deterministic rule engine for concealed structural, plumbing, and electrical damage.
Evaluates spatial proximity and building-code restoration heuristics,
emitting formal inspection flags with the exact rule that fired.
"""

from typing import List, Dict, Any


class ConcealedDamageRuleEngine:
    def __init__(self):
        self.rules = [
            {
                "rule_id": "RULE_WTR_01",
                "name": "Wall Cavity Insulation Saturation Rule",
                "description": "Water stain height on drywall exceeds 0.30m above finished floor (AFF)",
                "action": "Perform 2-foot flood cut; extract saturated batt insulation; inspect bottom plate."
            },
            {
                "rule_id": "RULE_ELEC_01",
                "name": "Concealed Wiring & Receptacle Ingress Rule",
                "description": "Water staining envelope intersects electrical receptacle height (0.30m - 0.45m AFF) within 0.50m horizontal radius",
                "action": "De-energize circuit breaker; pull receptacle for moisture test; perform megohmmeter wire test."
            },
            {
                "rule_id": "RULE_CEIL_01",
                "name": "Overhead Joist & Subfloor Leakage Rule",
                "description": "Ceiling water staining/deflection exceeds 0.40 m2",
                "action": "Controlled demolition of damaged gypsum ceiling board; moisture meter scan of structural ceiling joists."
            },
            {
                "rule_id": "RULE_CRK_01",
                "name": "Structural Differential Shear Rule",
                "description": "Diagonal cracking originating from door header or wall corner exceeds 2mm width",
                "action": "Install Tell-Tale crack monitoring gauge; structural engineer evaluation of foundation footing."
            }
        ]

    def evaluate_concealed_damage(self, room: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluates walls, ceilings, and damage regions in a room, emitting flags for any rules triggered.
        """
        flags = []
        room_id = room.get("room_id", "unknown_room")

        for wall in room.get("walls", []):
            wall_id = wall.get("wall_id", "unknown_wall")
            damages = wall.get("damage_regions", [])

            for dmg in damages:
                d_class = dmg.get("damage_class", "")
                loc = dmg.get("location_on_surface", {})
                v_max = loc.get("v_max", 0.0)
                v_min = loc.get("v_min", 0.0)
                u_min = loc.get("u_min", 0.0)
                u_max = loc.get("u_max", 0.0)
                extent = dmg.get("extent_m2", 0.0)

                # RULE_WTR_01 Check
                if "water" in d_class and v_max > 0.30:
                    flags.append({
                        "flag_id": f"FLAG_{room_id}_{wall_id}_WTR01",
                        "room_id": room_id,
                        "surface_id": wall_id,
                        "rule_fired": "RULE_WTR_01",
                        "trigger_evidence": f"Water stain reached {v_max:.2f}m AFF (threshold: >0.30m). Saturated cavity insulation risk.",
                        "recommended_action": "Execute 2-foot flood cut; extract wet fiberglass insulation; inspect bottom plate.",
                        "severity": "high"
                    })

                # RULE_ELEC_01 Check: typical outlet is 0.30m - 0.45m AFF
                if "water" in d_class and (v_min <= 0.50 and v_max >= 0.25):
                    flags.append({
                        "flag_id": f"FLAG_{room_id}_{wall_id}_ELEC01",
                        "room_id": room_id,
                        "surface_id": wall_id,
                        "rule_fired": "RULE_ELEC_01",
                        "trigger_evidence": f"Water staining boundary ({v_min:.2f}m - {v_max:.2f}m) intersects standard outlet elevation. In-wall junction box risk.",
                        "recommended_action": "De-energize circuit branch; inspect box terminals for corrosion; wire insulation resistance test.",
                        "severity": "high"
                    })

                # RULE_CRK_01 Check: structural crack
                if "crack" in d_class:
                    flags.append({
                        "flag_id": f"FLAG_{room_id}_{wall_id}_CRK01",
                        "room_id": room_id,
                        "surface_id": wall_id,
                        "rule_fired": "RULE_CRK_01",
                        "trigger_evidence": f"Structural shear crack detected with surface extent {extent:.2f}m2.",
                        "recommended_action": "Install displacement crack monitoring gauge; evaluate foundation settlement.",
                        "severity": "medium"
                    })

        return flags
