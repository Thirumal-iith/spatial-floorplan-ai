"""
benchmark/head_to_head.py
Part 3: our LiDAR-tier output vs. a consumer scanning app on 2 benchmark rooms.

Nothing in this file invents numbers. It needs three REAL inputs:
  1. benchmark/head_to_head/incumbent_export.json  - transcribed from the real app export
     (keep the original export file, e.g. PDF/CSV/JSON, next to it as evidence)
  2. benchmark/data/ground_truth.json             - laser/tape measurements
  3. our pipeline plan.json for the same capture  - produced by `python -m pipeline.run`

incumbent_export.json format:
{
  "app_name": "magicplan", "app_version": "x.y.z", "export_file": "magicplan_raw_export.pdf",
  "rooms": {
    "<room_id>": {
      "ceiling_height_m": 2.41, "floor_area_m2": 12.3,
      "walls_m": [4.10, 3.60, 4.10, 3.60],          # same order as ground_truth walls
      "openings_m": {"<opening_id>": 0.82}          # keyed like ground_truth openings
    }
  }
}
If the export is missing the gate is reported as NOT_RUN, never PASS.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, List, Optional

EXPORT_PATH = "benchmark/head_to_head/incumbent_export.json"
GT_PATH = "benchmark/data/ground_truth.json"
REPORT_PATH = "benchmark/head_to_head/head_to_head_report.md"


def _collect_dimensions(gt_room: Dict, inc_room: Dict, our_room: Dict) -> List[Dict[str, Any]]:
    dims = []

    def add(name, gt, inc, ours, unit="m"):
        if gt is None or inc is None or ours is None:
            return
        dims.append({"dim": name, "gt": float(gt), "inc": float(inc), "ours": float(ours), "unit": unit})

    add("Ceiling height", gt_room.get("ceiling_height_m"), inc_room.get("ceiling_height_m"),
        our_room["ceiling_height_m"]["val"])
    add("Floor area", gt_room.get("floor_area_m2"), inc_room.get("floor_area_m2"),
        our_room["floor_area_m2"]["val"], unit="m2")

    our_walls = our_room.get("walls", [])
    for i, gw in enumerate(gt_room.get("walls", [])):
        inc_w = inc_room.get("walls_m", [])
        if i < len(inc_w) and i < len(our_walls):
            add(f"Wall {i+1}", gw["length_m"], inc_w[i], our_walls[i]["length_m"]["val"])

    our_ops = {o["opening_id"]: o for w in our_walls for o in w.get("openings", [])}
    for gw in gt_room.get("walls", []):
        for gop in gw.get("openings", []):
            oid = gop.get("opening_id")
            ours = our_ops.get(oid)
            add(f"Opening {oid}", gop["width_m"], inc_room.get("openings_m", {}).get(oid),
                ours["width_m"]["val"] if ours else None)
    return dims


def run_head_to_head_comparison(our_plan_path: str) -> Optional[float]:
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    if not os.path.exists(EXPORT_PATH):
        msg = (f"# Part 3: Head-to-Head\n\n**Status: NOT_RUN** - no real incumbent export at `{EXPORT_PATH}`.\n"
               "Scan 2 benchmark rooms with Magicplan/Polycam free tier and transcribe the export.\n")
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(msg)
        print("[HEAD-TO-HEAD] NOT_RUN: incumbent export missing.")
        return None

    with open(EXPORT_PATH, encoding="utf-8") as f:
        inc = json.load(f)
    with open(GT_PATH, encoding="utf-8") as f:
        gt = json.load(f)
    with open(our_plan_path, encoding="utf-8") as f:
        ours = {r["room_id"]: r for r in json.load(f)["rooms"]}

    rows = []
    for rid, inc_room in inc["rooms"].items():
        if rid in gt["rooms"] and rid in ours:
            for d in _collect_dimensions(gt["rooms"][rid], inc_room, ours[rid]):
                d["room"] = rid
                rows.append(d)

    if not rows:
        print("[HEAD-TO-HEAD] No shared dimensions found.")
        return None

    wins = 0
    lines = [
        "# Part 3: Head-to-Head vs Incumbent App",
        f"**App:** {inc.get('app_name')} {inc.get('app_version')} (raw export: `{inc.get('export_file')}`)",
        "",
        "| Room | Dimension | Ground truth | Incumbent | Incumbent err | Ours | Our err | Result |",
        "| :-- | :-- | --: | --: | --: | --: | --: | :-: |",
    ]
    for r in rows:
        e_inc, e_our = abs(r["inc"] - r["gt"]), abs(r["ours"] - r["gt"])
        res = "BEAT" if e_our < e_inc else ("TIE" if e_our == e_inc else "LOSS")
        wins += res != "LOSS"
        scale, u = (100.0, "cm") if r["unit"] == "m" else (1.0, "m2")
        lines.append(f"| {r['room']} | {r['dim']} | {r['gt']:.3f} | {r['inc']:.3f} | {e_inc*scale:.2f} {u} "
                     f"| {r['ours']:.3f} | {e_our*scale:.2f} {u} | {res} |")
    rate = wins / len(rows) * 100.0
    gate = "PASS" if rate >= 70.0 else "FAIL"
    lines.insert(2, f"**Beat or tie:** {wins}/{len(rows)} ({rate:.1f}%) - threshold 70% - **{gate}**")
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[HEAD-TO-HEAD] {wins}/{len(rows)} ({rate:.1f}%) -> {gate}")
    return rate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="results/plan.json", help="Our LiDAR-tier plan.json")
    run_head_to_head_comparison(ap.parse_args().plan)
