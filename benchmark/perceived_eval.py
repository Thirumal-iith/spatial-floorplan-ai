"""Score the PERCEIVED video and photo tiers (raw RGB in, no scenario.json) against ground truth.

Rooms are matched to ground truth by nearest (sorted) bounding-box dimensions, since the
photo tier is not pose-placed. Writes benchmark/reports/perceived_results.json.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, ".")
GT = json.load(open("benchmark/data/ground_truth.json"))["rooms"]
OUT = "benchmark/reports/perceived_results.json"


def dims(poly):
    p = np.asarray(poly, float)
    return sorted((p.max(0) - p.min(0)).tolist(), reverse=True)


def n_openings(room, key="openings"):
    return sum(len(w.get(key, [])) for w in room["walls"])


def score(rooms, tol_pct):
    gt_dims = {k: dims(v["polygon"]) for k, v in GT.items()}
    used, rows = set(), []
    for r in rooms:
        d = dims(r["polygon"])
        cand = [(sum(abs(a - b) for a, b in zip(d, g)), k) for k, g in gt_dims.items() if k not in used]
        if not cand:
            continue
        _, k = min(cand)
        used.add(k)
        g = gt_dims[k]
        errs = [abs(a - b) / b * 100 for a, b in zip(d, g)]
        rows.append({
            "gt_room": k, "est_dims_m": [round(x, 3) for x in d], "gt_dims_m": g,
            "wall_err_pct": [round(e, 2) for e in errs],
            "ceiling_est_m": round(float(r["ceiling_height_m"]), 3),
            "ceiling_err_cm": round(abs(float(r["ceiling_height_m"]) - GT[k]["ceiling_height_m"]) * 100, 1),
            "openings_detected": n_openings(r), "openings_gt": n_openings(GT[k]),
            "convex_rectangle": len(r["polygon"]) == 4,
        })
    allerr = [e for row in rows for e in row["wall_err_pct"]]
    return {
        "rooms_reconstructed": len(rows), "rooms_gt": len(GT),
        "mean_wall_err_pct": round(float(np.mean(allerr)), 2) if allerr else None,
        "max_wall_err_pct": round(float(np.max(allerr)), 2) if allerr else None,
        "wall_gate_tol_pct": tol_pct,
        "wall_gate": "PASS" if allerr and max(allerr) <= tol_pct and len(rows) == len(GT) else "FAIL",
        "rooms": rows,
    }


def main():
    res = {"data_provenance": "SYNTHETIC_RENDERED_RGB", "evidence": "perceived"}
    from pipeline.geometry import video_layout as V
    t = time.time()
    rooms, _, _ = V.VideoReconstructor().reconstruct("benchmark/data/tier2_video_raw")
    res["video"] = score(rooms, 3.0)
    res["video"]["runtime_s"] = round(time.time() - t, 1)

    from pipeline.geometry import photo_layout as PL
    t = time.time()
    rooms, tel = PL.reconstruct_property("benchmark/data/tier1_photos_raw")
    res["photos"] = score(rooms, 8.0)
    res["photos"]["runtime_s"] = round(time.time() - t, 1)
    res["photos"]["failed_rooms"] = {k: v["error"] for k, v in tel.items() if "error" in v}

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w"), indent=2)
    for tier in ("video", "photos"):
        s = res[tier]
        print(f"[{tier.upper()} perceived] rooms {s['rooms_reconstructed']}/{s['rooms_gt']} "
              f"wall err mean {s['mean_wall_err_pct']}% max {s['max_wall_err_pct']}% "
              f"(gate +/-{s['wall_gate_tol_pct']}%: {s['wall_gate']}) runtime {s['runtime_s']}s")
        for row in s["rooms"]:
            print("   ", row["gt_room"], row["est_dims_m"], "vs", row["gt_dims_m"], row["wall_err_pct"],
                  f"ceil {row['ceiling_est_m']}", f"openings {row['openings_detected']}/{row['openings_gt']}")
    print("written", OUT)


if __name__ == "__main__":
    main()
