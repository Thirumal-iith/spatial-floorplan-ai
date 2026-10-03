"""
pipeline/rendering/plan_renderer.py
Vector architectural floor plan renderer.
Produces clean, publication-grade SVG floor plans (Polycam/Magicplan aesthetic)
with dimensioned walls, openings, color-coded multi-class damage overlays,
and professional title blocks conforming to REQ-16.
"""

from typing import Dict, Any, List
import math
import os


class FloorPlanRenderer:
    def __init__(self, canvas_size: int = 1200, padding: int = 120):
        self.canvas_size = canvas_size
        self.padding = padding

    def render_svg(self, stitched_plan: Dict[str, Any], output_path: str) -> str:
        """
        Renders the complete stitched floor plan to SVG.
        """
        svg_content = self._generate_svg_string(stitched_plan)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(svg_content)
        return output_path

    def _generate_svg_string(self, stitched_plan: Dict[str, Any]) -> str:
        rooms = stitched_plan.get("rooms", [])
        if not rooms:
            return '<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600"><text x="50" y="50" fill="white">No rooms found</text></svg>'

        # 1. Compute Bounding Box
        all_pts = []
        for r in rooms:
            poly = r.get("placed_polygon", r.get("polygon", []))
            all_pts.extend(poly)

        if not all_pts:
            return ""

        min_x = min(p[0] for p in all_pts)
        max_x = max(p[0] for p in all_pts)
        min_y = min(p[1] for p in all_pts)
        max_y = max(p[1] for p in all_pts)

        span_x = max(max_x - min_x, 1.0)
        span_y = max(max_y - min_y, 1.0)

        # Scale to canvas
        usable_w = self.canvas_size - 2 * self.padding
        usable_h = self.canvas_size - 2 * self.padding
        scale = min(usable_w / span_x, usable_h / span_y)

        def to_canvas(x: float, y: float):
            cx = self.padding + (x - min_x) * scale
            cy = self.canvas_size - (self.padding + (y - min_y) * scale)  # flip y for SVG
            return cx, cy

        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.canvas_size} {self.canvas_size}" width="{self.canvas_size}" height="{self.canvas_size}" style="background-color: #0f172a; font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif;">',
            '  <defs>',
            '    <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">',
            '      <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" stroke-width="1"/>',
            '    </pattern>',
            '    <pattern id="waterHatch" width="10" height="10" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">',
            '      <line x1="0" y1="0" x2="0" y2="10" stroke="#38bdf8" stroke-width="2.5" opacity="0.65"/>',
            '    </pattern>',
            '    <pattern id="crackHatch" width="8" height="8" patternTransform="rotate(-45 0 0)" patternUnits="userSpaceOnUse">',
            '      <line x1="0" y1="0" x2="0" y2="8" stroke="#ef4444" stroke-width="2.5" opacity="0.75"/>',
            '    </pattern>',
            '    <pattern id="moldHatch" width="8" height="8" patternUnits="userSpaceOnUse">',
            '      <circle cx="4" cy="4" r="2.2" fill="#10b981" opacity="0.7"/>',
            '    </pattern>',
            '  </defs>',
            '  <rect width="100%" height="100%" fill="url(#grid)" />'
        ]

        # 2. Draw Room Polygons & Fills
        for r in rooms:
            poly = r.get("placed_polygon", r.get("polygon", []))
            points_str = " ".join([f"{to_canvas(pt[0], pt[1])[0]:.1f},{to_canvas(pt[0], pt[1])[1]:.1f}" for pt in poly])
            svg_lines.append(f'  <polygon points="{points_str}" fill="#1e293b" fill-opacity="0.85" stroke="#64748b" stroke-width="2"/>')

        # 3. Draw Dimensioned Structural Walls & Openings
        for r in rooms:
            poly = r.get("placed_polygon", r.get("polygon", []))
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                c1 = to_canvas(p1[0], p1[1])
                c2 = to_canvas(p2[0], p2[1])

                # Heavy structural wall line
                svg_lines.append(f'  <line x1="{c1[0]:.1f}" y1="{c1[1]:.1f}" x2="{c2[0]:.1f}" y2="{c2[1]:.1f}" stroke="#f8fafc" stroke-width="6" stroke-linecap="round"/>')

                # Wall length dimension label
                wall_len = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                mid_x = (c1[0] + c2[0]) / 2.0
                mid_y = (c1[1] + c2[1]) / 2.0
                
                # Offset normal
                dx, dy = c2[0] - c1[0], c2[1] - c1[1]
                L = math.sqrt(dx*dx + dy*dy)
                if L > 1e-3:
                    nx, ny = -dy / L * 18, dx / L * 18
                    svg_lines.append(f'  <text x="{mid_x + nx:.1f}" y="{mid_y + ny:.1f}" fill="#94a3b8" font-size="11" font-weight="600" text-anchor="middle" dominant-baseline="middle">{wall_len:.2f}m</text>')

            # Room Center Label & Area
            c_x = sum(p[0] for p in poly) / n
            c_y = sum(p[1] for p in poly) / n
            cc_x, cc_y = to_canvas(c_x, c_y)
            r_name = r.get("name", "Room")
            area_val = r.get("floor_area_m2", {}).get("val", r.get("floor_area_m2", 0.0))
            ceiling_val = r.get("ceiling_height_m", {}).get("val", r.get("ceiling_height_m", 2.60))

            svg_lines.append(f'  <text x="{cc_x:.1f}" y="{cc_y - 12:.1f}" fill="#f1f5f9" font-size="16" font-weight="bold" text-anchor="middle">{r_name}</text>')
            svg_lines.append(f'  <text x="{cc_x:.1f}" y="{cc_y + 10:.1f}" fill="#38bdf8" font-size="13" font-weight="600" text-anchor="middle">{area_val:.2f} m²</text>')
            svg_lines.append(f'  <text x="{cc_x:.1f}" y="{cc_y + 28:.1f}" fill="#64748b" font-size="10" text-anchor="middle">H: {ceiling_val:.2f}m</text>')

        # 4. Accurate Multi-Class Damage Overlays & Leader Callouts
        for r in rooms:
            poly = r.get("placed_polygon", r.get("polygon", []))
            n = len(poly)
            if n < 3:
                continue
            r_center_x = sum(p[0] for p in poly) / n
            r_center_y = sum(p[1] for p in poly) / n

            for w_idx, wall in enumerate(r.get("walls", [])):
                p1 = poly[w_idx % n]
                p2 = poly[(w_idx + 1) % n]
                wall_len = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

                for dmg in wall.get("damage_regions", []):
                    d_class = dmg.get("damage_class", "").lower()
                    loc = dmg.get("location_on_surface", {})
                    u_min = loc.get("u_min", 0.5)
                    u_max = loc.get("u_max", 1.8)
                    u_mid = (u_min + u_max) / 2.0
                    t = max(0.15, min(0.85, u_mid / max(wall_len, 0.1)))

                    # Point along the wall in world coordinates
                    wx = p1[0] + t * (p2[0] - p1[0])
                    wy = p1[1] + t * (p2[1] - p1[1])
                    c_wx, c_wy = to_canvas(wx, wy)

                    # Vector pointing into room interior
                    vx = r_center_x - wx
                    vy = r_center_y - wy
                    dist = math.sqrt(vx*vx + vy*vy)
                    offset_m = 0.55
                    if dist > 1e-3:
                        cx_callout = wx + (vx / dist) * offset_m
                        cy_callout = wy + (vy / dist) * offset_m
                    else:
                        cx_callout, cy_callout = wx, wy
                    c_ox, c_oy = to_canvas(cx_callout, cy_callout)

                    # Extent metric value
                    raw_ext = dmg.get("extent_m2", 1.0)
                    ext_val = raw_ext.get("val", raw_ext) if isinstance(raw_ext, dict) else float(raw_ext)

                    # Style based on class
                    if "water" in d_class:
                        hatch_id = "waterHatch"
                        border_col = "#38bdf8"
                        bg_col = "#0c4a6e"
                        tag_txt = f"WATER STAIN: {ext_val:.2f} m²"
                        sub_txt = "IICRC S500 (Flood Cut 2ft)"
                    elif "crack" in d_class:
                        hatch_id = "crackHatch"
                        border_col = "#ef4444"
                        bg_col = "#450a0a"
                        lin_len = dmg.get("linear_extent_m", u_max - u_min)
                        tag_txt = f"SHEAR CRACK: {lin_len:.2f} m LF"
                        sub_txt = "ASTM E2126 (Epoxy Inject)"
                    elif "mold" in d_class:
                        hatch_id = "moldHatch"
                        border_col = "#10b981"
                        bg_col = "#064e3b"
                        tag_txt = f"MICROBIAL MOLD: {ext_val:.2f} m²"
                        sub_txt = "EPA Bio-Containment"
                    else:
                        hatch_id = "waterHatch"
                        border_col = "#f59e0b"
                        bg_col = "#451a03"
                        tag_txt = f"SURFACE DAMAGE: {ext_val:.2f} m²"
                        sub_txt = "Repair Required"

                    # 1. Wall Surface highlight line
                    svg_lines.append(f'  <line x1="{c_wx:.1f}" y1="{c_wy:.1f}" x2="{c_ox:.1f}" y2="{c_oy:.1f}" stroke="{border_col}" stroke-width="1.8" stroke-dasharray="3,3"/>')
                    
                    # 2. Damage Circle Marker
                    svg_lines.append(f'  <circle cx="{c_ox:.1f}" cy="{c_oy:.1f}" r="24" fill="url(#{hatch_id})" stroke="{border_col}" stroke-width="2.5"/>')
                    
                    # 3. Callout Box
                    box_w, box_h = 170, 36
                    bx = c_ox - box_w / 2.0
                    by = c_oy - 48
                    svg_lines.append(f'  <rect x="{bx:.1f}" y="{by:.1f}" width="{box_w}" height="{box_h}" rx="6" fill="{bg_col}" fill-opacity="0.95" stroke="{border_col}" stroke-width="1.5"/>')
                    svg_lines.append(f'  <text x="{c_ox:.1f}" y="{by + 15:.1f}" fill="#f8fafc" font-size="10.5" font-weight="bold" text-anchor="middle">{tag_txt}</text>')
                    svg_lines.append(f'  <text x="{c_ox:.1f}" y="{by + 28:.1f}" fill="{border_col}" font-size="8.5" font-weight="600" text-anchor="middle">{sub_txt}</text>')

        # 5. Professional Title Block (Magicplan style header)
        prop_id = stitched_plan.get("property_id", "Scan")
        tier = stitched_plan.get("tier", "lidar").upper()
        footprint = stitched_plan.get("total_footprint_m2", {}).get("val", stitched_plan.get("total_footprint_m2", 0.0))
        rest_cost = stitched_plan.get("total_estimated_restoration_cost", 0.0)

        num_rooms = len(rooms)
        if num_rooms == 1:
            r_single = rooms[0].get("name", prop_id)
            title_txt = f"SINGLE ROOM PLAN: {r_single}"
            mode_txt = f"MODE: SINGLE-ROOM SCAN (1 ROOM)  |  TIER: {tier}"
            rooms_txt = f'Total Footprint: <tspan fill="#f8fafc" font-weight="bold">{footprint:.2f} m²</tspan> (1 Isolated Room)'
        else:
            title_txt = f"MULTI-ROOM BLUEPRINT: {prop_id}"
            mode_txt = f"MODE: MULTI-ROOM GRAPH STITCHING ({num_rooms} ROOMS)  |  TIER: {tier}"
            rooms_txt = f'Total Footprint: <tspan fill="#f8fafc" font-weight="bold">{footprint:.2f} m²</tspan> ({num_rooms} Connected Rooms)'

        svg_lines.append('  <!-- Title Block -->')
        svg_lines.append('  <g transform="translate(40, 40)">')
        svg_lines.append('    <rect width="420" height="110" rx="8" fill="#1e293b" fill-opacity="0.95" stroke="#334155" stroke-width="1.5"/>')
        svg_lines.append(f'    <text x="20" y="32" fill="#f8fafc" font-size="15" font-weight="800">{title_txt}</text>')
        svg_lines.append(f'    <text x="20" y="56" fill="#38bdf8" font-size="11" font-weight="600">{mode_txt}</text>')
        svg_lines.append(f'    <text x="20" y="78" fill="#94a3b8" font-size="12">{rooms_txt}</text>')
        svg_lines.append(f'    <text x="20" y="98" fill="#f43f5e" font-size="12" font-weight="bold">Scope Estimate: ${rest_cost:,.2f}</text>')
        svg_lines.append('  </g>')

        # 6. Compass / North Arrow
        svg_lines.append('  <!-- North Arrow -->')
        svg_lines.append(f'  <g transform="translate({self.canvas_size - 80}, 60)">')
        svg_lines.append('    <circle cx="0" cy="0" r="24" fill="#1e293b" stroke="#334155" stroke-width="1.5"/>')
        svg_lines.append('    <path d="M 0 -16 L 6 8 L 0 4 L -6 8 Z" fill="#ef4444"/>')
        svg_lines.append('    <path d="M 0 16 L 6 8 L 0 4 L -6 8 Z" fill="#64748b"/>')
        svg_lines.append('    <text x="0" y="-20" fill="#f8fafc" font-size="10" font-weight="bold" text-anchor="middle">N</text>')
        svg_lines.append('  </g>')

        svg_lines.append('</svg>')
        return "\n".join(svg_lines)
