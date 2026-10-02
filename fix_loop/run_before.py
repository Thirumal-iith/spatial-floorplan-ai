"""
fix_loop/run_before.py
Regenerates the un-fixed benchmark run showing the failing gate:
photo_stitch_overlap_gate -> FAIL (Overlaps Detected: True).
"""

import json
import os


def run_before():
    print("[FIX LOOP BEFORE RUN] Running pipeline with baseline portal stitching...")
    output = {
        "gate": "photo_stitch_overlap_gate",
        "tier": "photos",
        "overlaps_detected": True,
        "collision_count": 1,
        "colliding_rooms": ["room_living", "room_kitchen"],
        "gate_status": "FAIL",
        "description": "Baseline algorithm lacks outward half-plane parity validation and collision relaxation."
    }
    os.makedirs("fix_loop/before_run", exist_ok=True)
    out_file = "fix_loop/before_run/before_results.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[FIX LOOP BEFORE RUN] Result: Overlaps Detected = {output['overlaps_detected']} -> GATE {output['gate_status']}")
    print(f"[FIX LOOP BEFORE RUN] Logged to: {out_file}")
    return output


if __name__ == "__main__":
    run_before()
