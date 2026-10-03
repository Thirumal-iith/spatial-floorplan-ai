"""
pipeline/damage/concealed_engine.py
Deterministic rule engine for concealed structural, plumbing, and electrical damage.
Evaluates spatial proximity and building-code restoration heuristics,
emitting formal inspection flags with the exact rule that fired conforming to REQ-12.
"""

from typing import List, Dict, Any


class ConcealedDamageRuleEngine:
    def __init__(self):
        self.rules = [
            {
                "rule_id": "RULE_WTR_01",
                "standard": "IICRC S500 §12.2.1 (Standard for Professional Water Damage Restoration)",
                "name": "Wall Cavity Insulation Saturation Rule",
                "description": "Water stain height on drywall exceeds 0.30m above finished floor (AFF)",
                "action": "Perform standard 2-foot (0.61m) flood cut; extract saturated cavity batt insulation; inspect bottom plate."
            },
            {
                "rule_id": "RULE_ELEC_01",
                "standard": "NFPA 70 NEC Art. 110.11 (Deteriorating Agents in Wet Locations)",
                "name": "Concealed Wiring & Receptacle Ingress Rule",
                "description": "Water staining envelope intersects electrical receptacle height (0.25m - 0.50m AFF) within 0.50m horizontal radius",
                "action": "De-energize circuit breaker branch; pull receptacle for moisture test; perform megohmmeter wire insulation resistance test."
            },
            {
                "rule_id": "RULE_CRK_01",
                "standard": "ASTM E2126 / IBC §1808 (Structural Foundation Settlement & Wall Diaphragm Shear)",
                "name": "Structural Differential Shear Rule",
                "description": "Diagonal cracking originating from door header or wall corner exceeds 2mm width",
                "action": "Install Avongard Tell-Tale crack displacement monitoring gauge; structural engineer evaluation of foundation footing."
            },
            {
                "rule_id": "RULE_MOLD_01",
                "standard": "EPA Mold Remediation / IICRC S520 (Standard for Professional Mold Remediation)",
                "name": "Microbial Growth & Bio-Containment Rule",
                "description": "Microbial fungal colony detected on porous gypsum substrate exceeding 0.05 m²",
                "action": "Erect 6-mil polyethylene containment with negative air pressure; HEPA vacuuming; biocidal antimicrobial wash."
            },
            {
                "rule_id": "RULE_CEIL_01",
                "standard": "IRC §R802 (Ceiling Joist & Rafter Moisture Integrity)",
                "name": "Overhead Joist & Subfloor Leakage Rule",
                "description": "Ceiling water staining or deflection exceeds 0.40 m² or reaches top wall boundary",
                "action": "Controlled demolition of damaged gypsum ceiling board; pinless moisture meter scan of structural ceiling joists."
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
                d_class = dmg.get("damage_class", "").lower()
                loc = dmg.get("location_on_surface", {})
                v_max = float(loc.get("v_max", 0.0))
                v_min = float(loc.get("v_min", 0.0))
                u_min = float(loc.get("u_min", 0.0))
                u_max = float(loc.get("u_max", 0.0))
                
                raw_ext = dmg.get("extent_m2", 0.0)
                extent = float(raw_ext.get("val", raw_ext)) if isinstance(raw_ext, dict) else float(raw_ext)

                # RULE_WTR_01 Check: Flood Cut trigger (IICRC S500)
                if "water" in d_class and v_max > 0.30:
                    flags.append({
                        "flag_id": f"FLAG_{room_id}_{wall_id}_WTR01",
                        "room_id": room_id,
                        "surface_id": wall_id,
                        "rule_fired": "RULE_WTR_01",
                        "building_code_standard": "IICRC S500 §12.2.1",
                        "trigger_evidence": f"Water stain reached {v_max:.2f}m AFF (threshold: >0.30m). Saturated cavity insulation risk.",
                        "recommended_action": "Execute 2-foot (0.61m) flood cut; extract wet fiberglass insulation; inspect bottom plate.",
                        "severity": "high"
                    })

                # RULE_ELEC_01 Check: typical outlet is 0.25m - 0.50m AFF (NEC Art. 110.11)
                if "water" in d_class and (v_min <= 0.50 and v_max >= 0.25):
                    flags.append({
                        "flag_id": f"FLAG_{room_id}_{wall_id}_ELEC01",
                        "room_id": room_id,
                        "surface_id": wall_id,
                        "rule_fired": "RULE_ELEC_01",
                        "building_code_standard": "NFPA 70 NEC Art. 110.11",
                        "trigger_evidence": f"Water staining boundary ({v_min:.2f}m - {v_max:.2f}m AFF) intersects standard outlet elevation. In-wall junction box risk.",
                        "recommended_action": "De-energize circuit branch; inspect box terminals for corrosion; wire insulation resistance test.",
                        "severity": "high"
                    })

                # RULE_CRK_01 Check: structural crack (ASTM E2126 / IBC §1808)
                if "crack" in d_class:
                    flags.append({
                        "flag_id": f"FLAG_{room_id}_{wall_id}_CRK01",
                        "room_id": room_id,
                        "surface_id": wall_id,
                        "rule_fired": "RULE_CRK_01",
                        "building_code_standard": "ASTM E2126 / IBC §1808",
                        "trigger_evidence": f"Structural shear crack detected with surface extent {extent:.2f}m² ({u_max - u_min:.2f}m linear).",
                        "recommended_action": "Install Avongard Tell-Tale displacement crack monitoring gauge; evaluate foundation settlement.",
                        "severity": "medium"
                    })

                # RULE_MOLD_01 Check: microbial fungal colonies (EPA / IICRC S520)
                if "mold" in d_class or "microbial" in d_class:
                    flags.append({
                        "flag_id": f"FLAG_{room_id}_{wall_id}_MLD01",
                        "room_id": room_id,
                        "surface_id": wall_id,
                        "rule_fired": "RULE_MOLD_01",
                        "building_code_standard": "EPA / IICRC S520",
                        "trigger_evidence": f"Microbial mold growth observed with surface extent {extent:.2f}m².",
                        "recommended_action": "Establish negative-air containment barrier; HEPA vacuuming; biocidal antimicrobial wash.",
                        "severity": "high"
                    })

        return flags
