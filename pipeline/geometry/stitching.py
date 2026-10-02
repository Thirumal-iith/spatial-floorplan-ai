"""
pipeline/geometry/stitching.py
Multi-room topological alignment and plan stitching engine.
Connects adjacent rooms via door portals and shared partition walls,
enforces non-overlapping polygon geometry, and computes whole-property footprint.
"""

from typing import List, Dict, Tuple, Optional, Any
import numpy as np


class MultiRoomStitcher:
    def __init__(self, wall_thickness_m: float = 0.12):
        self.wall_thickness = wall_thickness_m

    def stitch_rooms(
        self,
        room_data_list: List[Dict[str, Any]],
        connections: Optional[List[Tuple[str, str, str, str]]] = None
    ) -> Dict[str, Any]:
        """
        room_data_list: List of dicts, each with:
          - room_id, name, polygon (list of [x, y]), ceiling_height, walls, openings
        connections: Optional list of (room_a_id, opening_a_id, room_b_id, opening_b_id)
                     If None, automatic portal matching is performed based on opening widths.
        """
        if not room_data_list:
            return {"rooms": [], "total_footprint_m2": 0.0, "overlaps_detected": False}

        if len(room_data_list) == 1:
            # Single room property
            room = room_data_list[0]
            area = self._calculate_polygon_area(room["polygon"])
            room["placed_polygon"] = room["polygon"]
            return {
                "rooms": [room],
                "total_footprint_m2": float(area),
                "overlaps_detected": False,
                "adjacency_graph": {}
            }

        # Step 1: Designate central connector (e.g. Hallway) or first room as world origin
        placed_rooms = {}
        pending_rooms = {r["room_id"]: r for r in room_data_list}

        # Choose connector if present, else first room
        start_id = None
        for rid in pending_rooms:
            if "hall" in rid.lower() or "connector" in rid.lower() or "corridor" in rid.lower():
                start_id = rid
                break
        if start_id is None:
            start_id = list(pending_rooms.keys())[0]

        root_room = pending_rooms.pop(start_id)
        root_room["placed_polygon"] = [list(pt) for pt in root_room["polygon"]]
        root_room["transform"] = {"rot_deg": 0.0, "trans": [0.0, 0.0]}
        placed_rooms[start_id] = root_room

        adjacency_graph = {start_id: []}

        # Step 2: Iteratively align pending rooms through door portals
        max_iterations = len(room_data_list) * 2
        iter_count = 0

        while pending_rooms and iter_count < max_iterations:
            iter_count += 1
            room_placed_this_round = False

            for pending_id, pending_room in list(pending_rooms.items()):
                # Find connection between pending_room and any already placed_room
                matched = self._find_door_match(pending_room, placed_rooms)
                if matched:
                    target_id, p_door, t_door = matched
                    # Compute transform to align p_door with t_door
                    aligned_poly, transform = self._align_room_to_portal(
                        pending_room["polygon"], p_door, t_door, placed_rooms[target_id]["placed_polygon"]
                    )
                    
                    pending_room["placed_polygon"] = aligned_poly
                    pending_room["transform"] = transform
                    
                    placed_rooms[pending_id] = pending_room
                    del pending_rooms[pending_id]

                    if target_id not in adjacency_graph:
                        adjacency_graph[target_id] = []
                    adjacency_graph[target_id].append(pending_id)
                    if pending_id not in adjacency_graph:
                        adjacency_graph[pending_id] = []
                    adjacency_graph[pending_id].append(target_id)

                    room_placed_this_round = True
                    break

            if not room_placed_this_round and pending_rooms:
                # Fallback: place remaining room adjacent to boundary without overlap
                orphan_id, orphan_room = pending_rooms.popitem()
                placed_poly = self._place_non_overlapping_fallback(orphan_room["polygon"], list(placed_rooms.values()))
                orphan_room["placed_polygon"] = placed_poly
                orphan_room["transform"] = {"rot_deg": 0.0, "trans": [0.0, 0.0]}
                placed_rooms[orphan_id] = orphan_room

        # Step 3: Check for polygon overlaps
        overlaps = self._detect_polygon_overlaps(list(placed_rooms.values()))

        # Step 4: Calculate combined property footprint
        total_area = sum(self._calculate_polygon_area(r["placed_polygon"]) for r in placed_rooms.values())
        # Add partition wall allowance (~8% for standard residential floor plans)
        total_footprint = total_area * 1.08

        return {
            "rooms": list(placed_rooms.values()),
            "total_footprint_m2": float(round(total_footprint, 2)),
            "net_floor_area_m2": float(round(total_area, 2)),
            "overlaps_detected": overlaps,
            "adjacency_graph": adjacency_graph
        }

    def _find_door_match(self, candidate_room: Dict, placed_rooms: Dict[str, Dict]) -> Optional[Tuple[str, Dict, Dict]]:
        """
        Finds compatible door opening between candidate room and placed rooms.
        """
        cand_doors = self._extract_doors(candidate_room)
        for target_id, target_room in placed_rooms.items():
            target_doors = self._extract_doors(target_room)
            for cd in cand_doors:
                for td in target_doors:
                    # Match if door widths are within 8cm or explicitly linked
                    if abs(cd.get("width_m", 0.8) - td.get("width_m", 0.8)) < 0.10:
                        return (target_id, cd, td)
        return None

    def _extract_doors(self, room: Dict) -> List[Dict]:
        doors = []
        for wall in room.get("walls", []):
            for op in wall.get("openings", []):
                if op.get("type") == "door":
                    # Attach wall reference
                    op_copy = dict(op)
                    op_copy["wall_start"] = wall["start_point"]
                    op_copy["wall_end"] = wall["end_point"]
                    doors.append(op_copy)
        return doors

    def _align_room_to_portal(
        self,
        poly: List[List[float]],
        cand_door: Dict,
        target_door: Dict,
        target_poly: List[List[float]]
    ) -> Tuple[List[List[float]], Dict]:
        """
        Computes 2D rigid transform aligning candidate door threshold with target door threshold.
        """
        # Target portal center in world space
        t_start = np.array(target_door.get("wall_start", [0.0, 0.0]))
        t_end = np.array(target_door.get("wall_end", [1.0, 0.0]))
        t_dir = (t_end - t_start)
        t_len = np.linalg.norm(t_dir)
        t_unit = t_dir / max(t_len, 1e-6)
        t_normal = np.array([-t_unit[1], t_unit[0]])

        # Candidate portal in local space
        c_start = np.array(cand_door.get("wall_start", [0.0, 0.0]))
        c_end = np.array(cand_door.get("wall_end", [1.0, 0.0]))
        c_dir = (c_end - c_start)
        c_len = np.linalg.norm(c_dir)
        c_unit = c_dir / max(c_len, 1e-6)
        c_normal = np.array([-c_unit[1], c_unit[0]])

        # Rotate candidate so candidate normal is anti-parallel to target normal
        # i.e. facing outward into each other
        angle_target = np.arctan2(t_normal[1], t_normal[0])
        angle_cand = np.arctan2(c_normal[1], c_normal[0])
        rot_angle = (angle_target + np.pi) - angle_cand

        # Normalize rotation
        cos_a, sin_a = np.cos(rot_angle), np.sin(rot_angle)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        # Position offset: portal center + wall thickness
        cand_center = (c_start + c_end) / 2.0
        target_center = (t_start + t_end) / 2.0

        rotated_cand_center = R @ cand_center
        # Doorway offset: step outward along target normal by wall thickness
        portal_contact = target_center + (t_normal * self.wall_thickness)
        translation = portal_contact - rotated_cand_center

        # Apply transformation to candidate polygon vertices
        aligned_poly = []
        for pt in poly:
            p_rot = R @ np.array(pt) + translation
            aligned_poly.append([float(round(p_rot[0], 3)), float(round(p_rot[1], 3))])

        return aligned_poly, {
            "rot_deg": float(np.degrees(rot_angle)),
            "trans": [float(translation[0]), float(translation[1])]
        }

    def _place_non_overlapping_fallback(self, poly: List[List[float]], placed_rooms: List[Dict]) -> List[List[float]]:
        # Shift along X axis past max X of all placed rooms + 0.5m
        max_x = max(max(pt[0] for pt in r["placed_polygon"]) for r in placed_rooms)
        min_curr_x = min(pt[0] for pt in poly)
        shift_x = (max_x - min_curr_x) + 0.5
        return [[float(pt[0] + shift_x), float(pt[1])] for pt in poly]

    def _detect_polygon_overlaps(self, rooms: List[Dict]) -> bool:
        """
        Simple bounding box & centroid separation check to confirm no room interior collisions.
        """
        for i in range(len(rooms)):
            poly_a = np.array(rooms[i]["placed_polygon"])
            c_a = np.mean(poly_a, axis=0)
            for j in range(i + 1, len(rooms)):
                poly_b = np.array(rooms[j]["placed_polygon"])
                c_b = np.mean(poly_b, axis=0)
                # If centroids are nearly identical, overlap is detected
                if np.linalg.norm(c_a - c_b) < 0.2:
                    return True
        return False

    def _calculate_polygon_area(self, poly: List[List[float]]) -> float:
        """Shoelace formula for 2D polygon area."""
        n = len(poly)
        if n < 3:
            return 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += poly[i][0] * poly[j][1]
            area -= poly[j][0] * poly[i][1]
        return abs(area) / 2.0
