"""
pipeline/damage/scoping.py
Insurance restoration scope line item generator.
Converts surface damage extents and concealed-damage flags into standardized
restoration line items keyed to specific physical surfaces.
"""

from typing import List, Dict, Any


class RestorationScoper:
    # Standard restoration cost database (Labor + Materials in USD)
    UNIT_RATES = {
        "WTR_FLOOD_CUT_2FT": {"code": "WTR-DRY-CUT", "unit": "LF", "rate": 18.50, "desc": "Tear out wet drywall up to 2-foot flood cut, bag & haul debris"},
        "INS_REPLACE_R13": {"code": "INS-BAT-R13", "unit": "SF", "rate": 2.85, "desc": "Replace wet R-13 fiberglass batt wall cavity insulation"},
        "DRY_HANG_FINISH": {"code": "DRY-HANG-F", "unit": "SF", "rate": 4.20, "desc": "Hang, tape, float, sand and finish drywall"},
        "PNT_PRIME_PAINT": {"code": "PNT-PR-W", "unit": "SF", "rate": 2.15, "desc": "Apply antimicrobial stain-blocking primer and two coats acrylic latex"},
        "ELEC_REPLACE_DEVICE": {"code": "ELEC-SWT-R", "unit": "EA", "rate": 65.00, "desc": "Test branch wiring and replace water-exposed electrical duplex receptacle"},
        "ANTIMICROBIAL_SPRAY": {"code": "WTR-MOLD-MED", "unit": "SF", "rate": 1.10, "desc": "Apply EPA-registered antimicrobial agent to exposed wall studs"},
        "CRACK_EPOXY_INJECT": {"code": "MAS-CRK-REP", "unit": "LF", "rate": 45.00, "desc": "Stitch and inject structural epoxy into shear crack"}
    }

    def generate_scope_for_room(
        self,
        room: Dict[str, Any],
        concealed_flags: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generates line items for all visible damage and triggered concealed flags in a room.
        Keyed to exact surfaces (e.g. room_living/wall_01).
        """
        items = []
        room_id = room.get("room_id", "room")
        counter = 1

        for wall in room.get("walls", []):
            wall_id = wall.get("wall_id", "wall")
            surface_ref = f"{room_id}/{wall_id}"
            damages = wall.get("damage_regions", [])
            wall_len_m = wall.get("length_m", 3.0)

            for dmg in damages:
                d_class = dmg.get("damage_class", "")
                extent_m2 = dmg.get("extent_m2", 1.0)
                # Convert m2 to Square Feet (1 m2 = 10.7639 sq ft)
                sf = round(extent_m2 * 10.7639, 1)
                # Convert linear meters to Linear Feet (1 m = 3.28084 ft)
                lf = round(min(wall_len_m, extent_m2 * 1.5) * 3.28084, 1)

                if "water" in d_class:
                    # 1. Flood cut line item
                    r = self.UNIT_RATES["WTR_FLOOD_CUT_2FT"]
                    items.append({
                        "item_id": f"SC_{room_id}_{counter:03d}",
                        "surface_ref": surface_ref,
                        "category": "WTR",
                        "code": r["code"],
                        "description": r["desc"],
                        "quantity": lf,
                        "unit": r["unit"],
                        "estimated_unit_cost": r["rate"],
                        "total_cost": round(lf * r["rate"], 2)
                    })
                    counter += 1

                    # 2. Insulation replacement
                    r = self.UNIT_RATES["INS_REPLACE_R13"]
                    items.append({
                        "item_id": f"SC_{room_id}_{counter:03d}",
                        "surface_ref": surface_ref,
                        "category": "INS",
                        "code": r["code"],
                        "description": r["desc"],
                        "quantity": sf,
                        "unit": r["unit"],
                        "estimated_unit_cost": r["rate"],
                        "total_cost": round(sf * r["rate"], 2)
                    })
                    counter += 1

                    # 3. Antimicrobial remediation
                    r = self.UNIT_RATES["ANTIMICROBIAL_SPRAY"]
                    items.append({
                        "item_id": f"SC_{room_id}_{counter:03d}",
                        "surface_ref": surface_ref,
                        "category": "WTR",
                        "code": r["code"],
                        "description": r["desc"],
                        "quantity": sf,
                        "unit": r["unit"],
                        "estimated_unit_cost": r["rate"],
                        "total_cost": round(sf * r["rate"], 2)
                    })
                    counter += 1

                    # 4. Drywall hang and paint
                    r_dry = self.UNIT_RATES["DRY_HANG_FINISH"]
                    items.append({
                        "item_id": f"SC_{room_id}_{counter:03d}",
                        "surface_ref": surface_ref,
                        "category": "DRY",
                        "code": r_dry["code"],
                        "description": r_dry["desc"],
                        "quantity": sf,
                        "unit": r_dry["unit"],
                        "estimated_unit_cost": r_dry["rate"],
                        "total_cost": round(sf * r_dry["rate"], 2)
                    })
                    counter += 1

                    r_pnt = self.UNIT_RATES["PNT_PRIME_PAINT"]
                    items.append({
                        "item_id": f"SC_{room_id}_{counter:03d}",
                        "surface_ref": surface_ref,
                        "category": "PNT",
                        "code": r_pnt["code"],
                        "description": r_pnt["desc"],
                        "quantity": sf,
                        "unit": r_pnt["unit"],
                        "estimated_unit_cost": r_pnt["rate"],
                        "total_cost": round(sf * r_pnt["rate"], 2)
                    })
                    counter += 1

                elif "crack" in d_class:
                    r = self.UNIT_RATES["CRACK_EPOXY_INJECT"]
                    items.append({
                        "item_id": f"SC_{room_id}_{counter:03d}",
                        "surface_ref": surface_ref,
                        "category": "STR",
                        "code": r["code"],
                        "description": r["desc"],
                        "quantity": lf,
                        "unit": r["unit"],
                        "estimated_unit_cost": r["rate"],
                        "total_cost": round(lf * r["rate"], 2)
                    })
                    counter += 1

        # Check for electrical concealed damage flags
        for flag in concealed_flags:
            if flag.get("rule_fired") == "RULE_ELEC_01":
                r = self.UNIT_RATES["ELEC_REPLACE_DEVICE"]
                items.append({
                    "item_id": f"SC_{room_id}_{counter:03d}",
                    "surface_ref": f"{room_id}/{flag.get('surface_id')}",
                    "category": "ELEC",
                    "code": r["code"],
                    "description": r["desc"],
                    "quantity": 1.0,
                    "unit": r["unit"],
                    "estimated_unit_cost": r["rate"],
                    "total_cost": r["rate"]
                })
                counter += 1

        return items
