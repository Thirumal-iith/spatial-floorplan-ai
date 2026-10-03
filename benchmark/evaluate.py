"""
benchmark/evaluate.py

Scores pipeline outputs against ground truth. Every number is computed from pipeline runs.

Gates (case study Part 2):
  LiDAR   openings <= 2 cm on >= 85% (missed AND phantom openings count as misses)
          ceiling <= 1.5 cm per room; spread across repeat captures <= 1 cm; bias vs unrepeatable stated
          repeatability: same room twice agrees within 1 cm or 0.5% per wall
  Drift   ablation ON vs OFF on the multi-room capture
  Video   walls within 3%      Photos  walls within 8%, stitched footprint within 8%, no overlaps
  Calibration: fraction of ground-truth values inside the reported 95% CI, at every tier.

Each tier result carries `evidence`: "perceived" (geometry measured from sensor data) or
"passthrough" (geometry read from scenario.json; the gate only proves the plumbing works).

Rooms/walls/openings are matched by id when the pipeline used GT ids (passthrough tiers),
otherwise geometrically in the world frame (perceived tiers).
"""

import os
import sys
import json
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pipeline.run import PipelineRunner  # noqa: E402
from pipeline.geometry.stitching import MultiRoomStitcher  # noqa: E402

DATA = "benchmark/data"
OPENING_MATCH_RADIUS_M = 0.40


def _val(x):
    return x["val"] if isinstance(x, dict) else float(x)


def _in_ci(x, gt) -> Optional[bool]:
    if isinstance(x, dict) and "ci_95" in x:
        lo, hi = x["ci_95"]
        return lo <= gt <= hi
    return None


class BenchmarkEvaluator:
    def __init__(self, ground_truth_path: str = f"{DATA}/ground_truth.json"):
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            self.gt = json.load(f)
        self._stitch = MultiRoomStitcher()

    # ================================================================ orchestration
    def evaluate_all_tiers(self, data_dir: str = DATA) -> Dict[str, Any]:
        on = PipelineRunner(drift_correction=True)
        off = PipelineRunner(drift_correction=False)
        results: Dict[str, Any] = {"ground_truth_provenance": self.gt.get("data_provenance", "unknown")}

        lidar_dir = os.path.join(data_dir, "tier3_lidar_raw")
        lidar = on.process_capture(lidar_dir, tier="lidar")
        results["lidar"] = self._score_tier(lidar, "lidar")

        repeat = on.process_capture(os.path.join(data_dir, "tier3_lidar_raw_bedroom_repeat"), tier="lidar")
        results["repeatability"] = self._score_repeatability(lidar, repeat, "room_bedroom")

        video = on.process_capture(os.path.join(data_dir, "tier2_video"), tier="video")
        results["video"] = self._score_tier(video, "video")

        photos = on.process_capture(os.path.join(data_dir, "tier1_photos"), tier="photos")
        results["photos"] = self._score_tier(photos, "photos")

        lidar_off = off.process_capture(lidar_dir, tier="lidar")
        results["drift_ablation"] = self._score_drift_ablation(lidar, lidar_off)

        results["damage_evaluation"] = self._score_damage_evaluation(video)
        results["timing_s"] = {"lidar": lidar["processing_time_seconds"], "video": video["processing_time_seconds"],
                               "photos": photos["processing_time_seconds"]}
        return results

    # ================================================================ matching
    @staticmethod
    def _evidence(plan) -> str:
        return "perceived" if str(plan.get("data_provenance", "")).startswith("perceived") else "passthrough"

    def _match_rooms(self, plan) -> Dict[str, Dict]:
        preds = plan.get("rooms", [])
        by_id = {p["room_id"]: p for p in preds if p["room_id"] in self.gt["rooms"]}
        if by_id:
            return by_id
        mapping = {}
        for gid, g in self.gt["rooms"].items():
            best, best_o = None, 0.0
            for p in preds:
                union, overlap = self._stitch.union_and_overlap([p["polygon"], g["world_polygon"]])
                if overlap > best_o:
                    best, best_o = p, overlap
            if best is not None and best_o > 0.5 * g["floor_area_m2"]:
                mapping[gid] = best
        return mapping

    @staticmethod
    def _seg(w) -> Tuple[np.ndarray, np.ndarray]:
        a = np.asarray(w.get("placed_start_point") or w["start_point"], dtype=float)
        b = np.asarray(w.get("placed_end_point") or w["end_point"], dtype=float)
        return a, b

    def _match_walls(self, gid: str, pred_room: Dict) -> List[Tuple[Dict, Optional[Dict]]]:
        g_room = self.gt["rooms"][gid]
        pw = pred_room.get("walls", [])
        if pred_room["room_id"] == gid:
            by_id = {w["wall_id"]: w for w in pw}
            return [(gw, by_id.get(gw["wall_id"])) for gw in g_room["walls"]]
        pairs = []
        for gw in g_room["walls"]:
            ga, gb = np.asarray(gw["world_start"], float), np.asarray(gw["world_end"], float)
            gd = (gb - ga) / np.linalg.norm(gb - ga)
            gm = (ga + gb) / 2
            best, bd = None, 0.5
            for w in pw:
                a, b = self._seg(w)
                L = np.linalg.norm(b - a)
                if L < 1e-6 or abs(((b - a) / L) @ gd) < 0.95:
                    continue
                d = np.linalg.norm((a + b) / 2 - gm)
                if d < bd:
                    best, bd = w, d
            pairs.append((gw, best))
        return pairs

    def _match_openings(self, gw: Dict, pw: Optional[Dict], by_id: bool):
        """Returns (matched [(gt_op, pred_op)], missed [gt_op], phantom [pred_op])."""
        gops = gw.get("openings", [])
        pops = list(pw.get("openings", [])) if pw else []
        matched, missed = [], []
        if by_id:
            pid = {o["opening_id"]: o for o in pops}
            for g in gops:
                p = pid.pop(g["opening_id"], None)
                (matched.append((g, p)) if p else missed.append(g))
            return matched, missed, list(pid.values())
        a, b = self._seg(pw) if pw else (None, None)
        remaining = pops[:]
        for g in gops:
            best, bd = None, OPENING_MATCH_RADIUS_M
            for p in remaining:
                L = np.linalg.norm(b - a)
                c = a + (b - a) / L * p["offset_along_wall_m"]
                d = np.linalg.norm(c - np.asarray(g["center_world"]))
                if d < bd:
                    best, bd = p, d
            if best is not None:
                remaining.remove(best)
                matched.append((g, best))
            else:
                missed.append(g)
        return matched, missed, remaining

    # ================================================================ tier scoring
    def _score_tier(self, plan: Dict[str, Any], tier: str) -> Dict[str, Any]:
        mapping = self._match_rooms(plan)
        wall_err_pct, wall_err_cm, ceil_err_cm, area_err_pct = [], [], [], []
        op_err_cm, n_missed, n_phantom = [], 0, 0
        ci_hits: Dict[str, List[bool]] = {"walls": [], "ceilings": [], "openings": [], "room_areas": []}
        missing_walls = 0

        for gid, g in self.gt["rooms"].items():
            p = mapping.get(gid)
            if p is None:
                missing_walls += len(g["walls"])
                n_missed += sum(len(w.get("openings", [])) for w in g["walls"])
                continue
            ceil_err_cm.append(abs(_val(p["ceiling_height_m"]) - g["ceiling_height_m"]) * 100)
            ci_hits["ceilings"].append(bool(_in_ci(p["ceiling_height_m"], g["ceiling_height_m"])))
            area_err_pct.append(abs(_val(p["floor_area_m2"]) - g["floor_area_m2"]) / g["floor_area_m2"] * 100)
            ci_hits["room_areas"].append(bool(_in_ci(p["floor_area_m2"], g["floor_area_m2"])))
            by_id = p["room_id"] == gid
            matched_walls = set()
            for gw, pw in self._match_walls(gid, p):
                if pw is None:
                    missing_walls += 1
                    n_missed += len(gw.get("openings", []))
                    continue
                matched_walls.add(pw["wall_id"])
                L = _val(pw["length_m"])
                wall_err_cm.append(abs(L - gw["length_m"]) * 100)
                wall_err_pct.append(abs(L - gw["length_m"]) / gw["length_m"] * 100)
                ci_hits["walls"].append(bool(_in_ci(pw["length_m"], gw["length_m"])))
                m, miss, ph = self._match_openings(gw, pw, by_id)
                for go, po in m:
                    op_err_cm.append(abs(_val(po["width_m"]) - go["width_m"]) * 100)
                    ci_hits["openings"].append(bool(_in_ci(po["width_m"], go["width_m"])))
                n_missed += len(miss)
                n_phantom += len(ph)
            # openings on predicted walls that matched no GT wall are phantoms too
            n_phantom += sum(len(w.get("openings", [])) for w in p.get("walls", [])
                             if w["wall_id"] not in matched_walls)

        n_ok = sum(1 for e in op_err_cm if e <= 2.0)
        denom = len(op_err_cm) + n_missed + n_phantom
        pct_ok = n_ok / denom * 100 if denom else 0.0
        gt_fp = self.gt["total_ground_truth_footprint_m2"]
        fp_err_pct = abs(_val(plan["total_footprint_m2"]) - gt_fp) / gt_fp * 100
        cov = {k: (round(100.0 * sum(v) / len(v), 1) if v else None) for k, v in ci_hits.items()}
        all_hits = [h for v in ci_hits.values() for h in v]
        cov["all"] = round(100.0 * sum(all_hits) / len(all_hits), 1) if all_hits else None
        mean_w = float(np.mean(wall_err_pct)) if wall_err_pct else float("nan")
        max_w = float(np.max(wall_err_pct)) if wall_err_pct else float("nan")
        max_c = float(np.max(ceil_err_cm)) if ceil_err_cm else float("nan")

        gates = {}
        if tier == "lidar":
            gates["opening_widths_gate"] = "PASS" if pct_ok >= 85.0 else "FAIL"
            gates["ceiling_height_gate"] = "PASS" if max_c <= 1.5 else "FAIL"
        elif tier == "video":
            gates["wall_accuracy_gate"] = "PASS" if max_w <= 3.0 else "FAIL"
        elif tier == "photos":
            gates["wall_accuracy_gate"] = "PASS" if max_w <= 8.0 else "FAIL"
            gates["photo_stitch_overlap_gate"] = "PASS" if not plan.get("overlaps_detected") else "FAIL"
            gates["footprint_within_8pct_gate"] = "PASS" if fp_err_pct <= 8.0 else "FAIL"
            adj_ok = self._adjacency_ok(plan, mapping)
            gates["adjacency_gate"] = "PASS" if adj_ok else "FAIL"
        gates["calibration_gate"] = "PASS" if (cov["all"] or 0) >= 95.0 else "FAIL"

        return {
            "tier": tier,
            "evidence": self._evidence(plan),
            "rooms_matched": f"{len(mapping)}/{len(self.gt['rooms'])}",
            "walls_scored": len(wall_err_pct), "walls_missing": missing_walls,
            "wall_error_mean_pct": round(mean_w, 3), "wall_error_max_pct": round(max_w, 3),
            "wall_error_mean_cm": round(float(np.mean(wall_err_cm)), 2) if wall_err_cm else None,
            "openings_matched": len(op_err_cm), "openings_missed": n_missed, "openings_phantom": n_phantom,
            "opening_error_mean_cm": round(float(np.mean(op_err_cm)), 2) if op_err_cm else None,
            "pct_openings_le_2cm": round(pct_ok, 1),
            "max_ceiling_error_cm": round(max_c, 2),
            "room_area_error_mean_pct": round(float(np.mean(area_err_pct)), 2) if area_err_pct else None,
            "footprint_error_pct": round(fp_err_pct, 3),
            "overlaps_detected": plan.get("overlaps_detected", False),
            "ci95_coverage_pct": cov,
            "gates": gates,
        }

    def _adjacency_ok(self, plan, mapping) -> bool:
        """Rooms connected by a door in GT must touch (within 0.3 m) in the stitched plan."""
        inv = {p["room_id"]: gid for gid, p in mapping.items()}
        polys = {gid: np.asarray(p.get("placed_polygon") or p["polygon"], float) for gid, p in mapping.items()}
        for gid, g in self.gt["rooms"].items():
            for w in g["walls"]:
                for o in w.get("openings", []):
                    other = o.get("connected_room")
                    if other and gid in polys and other in polys:
                        a, b = polys[gid], polys[other]
                        d = min(np.min(np.linalg.norm(a[:, None] - b[None], axis=2)),
                                self._poly_gap(a, b))
                        if d > 0.3:
                            return False
        return len(mapping) == len(self.gt["rooms"])

    @staticmethod
    def _poly_gap(a, b) -> float:
        amin, amax, bmin, bmax = a.min(0), a.max(0), b.min(0), b.max(0)
        gap = np.maximum(0, np.maximum(bmin - amax, amin - bmax))
        return float(np.linalg.norm(gap))

    # ================================================================ repeatability
    def _score_repeatability(self, run1: Dict, run2: Dict, gid: str) -> Dict[str, Any]:
        r1, r2 = self._match_rooms(run1).get(gid), self._match_rooms(run2).get(gid)
        if not r1 or not r2:
            return {"room_id": gid, "overall_repeatability_gate": "FAIL", "reason": "room not found in both captures"}
        w1 = {gw["wall_id"]: pw for gw, pw in self._match_walls(gid, r1)}
        w2 = {gw["wall_id"]: pw for gw, pw in self._match_walls(gid, r2)}
        rows, ok_all = [], True
        for gw in self.gt["rooms"][gid]["walls"]:
            a, b = w1.get(gw["wall_id"]), w2.get(gw["wall_id"])
            if not a or not b:
                ok_all = False
                rows.append({"wall": gw["wall_id"], "status": "MISSING"})
                continue
            l1, l2 = _val(a["length_m"]), _val(b["length_m"])
            dcm, dpct = abs(l1 - l2) * 100, abs(l1 - l2) / gw["length_m"] * 100
            ok = dcm <= 1.0 or dpct <= 0.5
            ok_all &= ok
            rows.append({"wall": gw["wall_id"], "gt_m": gw["length_m"], "capture1_m": l1, "capture2_m": l2,
                         "diff_cm": round(dcm, 2), "diff_pct": round(dpct, 3), "status": "PASS" if ok else "FAIL"})
        g_h = self.gt["rooms"][gid]["ceiling_height_m"]
        h1, h2 = _val(r1["ceiling_height_m"]), _val(r2["ceiling_height_m"])
        spread = abs(h1 - h2) * 100
        bias = ((h1 + h2) / 2 - g_h) * 100
        if spread > 1.0:
            cls = "unrepeatable"
        elif abs(bias) > 1.5:
            cls = "repeatable but biased"
        else:
            cls = "repeatable and unbiased"
        return {
            "room_id": gid, "evidence": self._evidence(run1), "walls": rows,
            "max_wall_diff_cm": max((r["diff_cm"] for r in rows if "diff_cm" in r), default=None),
            "max_wall_diff_pct": max((r["diff_pct"] for r in rows if "diff_pct" in r), default=None),
            "ceiling_capture1_m": h1, "ceiling_capture2_m": h2,
            "ceiling_spread_cm": round(spread, 2), "ceiling_mean_bias_cm": round(bias, 2),
            "classification": cls,
            "wall_agreement_gate": "PASS" if ok_all else "FAIL",
            "ceiling_spread_gate": "PASS" if spread <= 1.0 else "FAIL",
            "overall_repeatability_gate": "PASS" if ok_all and cls == "repeatable and unbiased" else "FAIL",
        }

    # ================================================================ drift ablation
    def _score_drift_ablation(self, run_on: Dict, run_off: Dict) -> Dict[str, Any]:
        s_on, s_off = self._score_tier(run_on, "lidar"), self._score_tier(run_off, "lidar")
        m_on, m_off = run_on.get("drift_metrics", {}), run_off.get("drift_metrics", {})

        def side(run, s, m):
            pe = self._placement_errors(run)
            return {
                "rooms_matched": s["rooms_matched"], "walls_scored": s["walls_scored"],
                "walls_missing": s["walls_missing"],
                "wall_error_mean_pct": s["wall_error_mean_pct"], "wall_error_max_pct": s["wall_error_max_pct"],
                "footprint_m2": _val(run["total_footprint_m2"]), "footprint_error_pct": s["footprint_error_pct"],
                "overlap_area_m2": run.get("overlap_area_m2"),
                "pct_openings_le_2cm": s["pct_openings_le_2cm"],
                "loop_closure_gap_cm": None if m.get("residual_drift_m") is None
                else round(m["residual_drift_m"] * 100, 2),
                **pe,
            }

        on, off = side(run_on, s_on, m_on), side(run_off, s_off, m_off)
        corrected = bool(m_on.get("drift_correction_applied"))
        better = (on["mean_centroid_err_cm"] is not None and
                  (off["mean_centroid_err_cm"] is None or on["mean_centroid_err_cm"] < off["mean_centroid_err_cm"])
                  and on["wall_error_mean_pct"] <= off["wall_error_mean_pct"])
        return {
            "method": m_on.get("method"), "loop_closures": m_on.get("loop_closures_found"),
            "manhattan_anchors": m_on.get("manhattan_anchors_used"),
            "max_yaw_correction_deg": round(m_on.get("max_yaw_correction_deg", 0.0), 3),
            "drift_correction_ON": on, "drift_correction_OFF": off,
            "gate": "PASS" if corrected and better else "FAIL",
        }

    def _placement_errors(self, plan: Dict) -> Dict[str, Any]:
        mapping = self._match_rooms(plan)
        cerr = []
        for gid, p in mapping.items():
            g = np.asarray(self.gt["rooms"][gid]["world_polygon"], float)
            q = np.asarray(p["polygon"], float)
            cerr.append(np.linalg.norm(q.mean(0) - g.mean(0)) * 100)
        if not cerr:
            return {"mean_centroid_err_cm": None, "max_centroid_err_cm": None}
        return {"mean_centroid_err_cm": round(float(np.mean(cerr)), 2),
                "max_centroid_err_cm": round(float(np.max(cerr)), 2)}

    # ================================================================ damage
    def _score_damage_evaluation(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        gt_damages = self.gt["rooms"]["room_living"].get("staged_damages", [])
        detected = [d for r in plan.get("rooms", []) for w in r.get("walls", []) for d in w.get("damage_regions", [])]
        classes_gt = {d["damage_class"] for d in gt_damages}
        classes_pred = {d["damage_class"] for d in detected}
        hits, errs = 0, []
        for g in gt_damages:
            m = next((d for d in detected if d["damage_class"] == g["damage_class"]), None)
            if m:
                errs.append(abs(_val(m["extent_m2"]) - g["extent_m2"]) / g["extent_m2"] * 100)
                hits += bool(_in_ci(m["extent_m2"], g["extent_m2"]))
        rules = sorted({f["rule_fired"] for f in plan.get("concealed_damage_flags", [])})
        scope = plan.get("scope_line_items", [])
        evidence = self._evidence(plan)
        return {
            "evidence": evidence,
            "note": "Damage regions are copied from ground truth in passthrough mode; detector not yet run on images."
            if evidence == "passthrough" else "",
            "classes_detected": sorted(classes_pred),
            "multi_class_gate": "PASS" if classes_gt <= classes_pred and len(classes_pred) >= 2 else "FAIL",
            "extent_ci95_coverage_pct": round(100.0 * hits / max(len(gt_damages), 1), 1),
            "mean_extent_error_pct": round(float(np.mean(errs)), 2) if errs else None,
            "rules_fired": rules,
            "scope_line_items_count": len(scope),
        }


def _fmt_gates(d):
    return ", ".join(f"{k}={v}" for k, v in d.items())


def main():
    ev = BenchmarkEvaluator()
    rep = ev.evaluate_all_tiers()
    os.makedirs("benchmark/reports", exist_ok=True)
    with open("benchmark/reports/benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, default=float)

    print("=" * 72)
    print(f" BENCHMARK (ground truth provenance: {rep['ground_truth_provenance'].upper()})")
    print("=" * 72)
    for t in ("lidar", "video", "photos"):
        r = rep[t]
        print(f"\n[{t.upper()}] evidence={r['evidence']} rooms={r['rooms_matched']} walls={r['walls_scored']} "
              f"(missing {r['walls_missing']})")
        print(f"  walls: mean {r['wall_error_mean_pct']}% ({r['wall_error_mean_cm']} cm), max {r['wall_error_max_pct']}%")
        print(f"  openings: {r['openings_matched']} matched, {r['openings_missed']} missed, "
              f"{r['openings_phantom']} phantom; <=2cm: {r['pct_openings_le_2cm']}%; mean err {r['opening_error_mean_cm']} cm")
        print(f"  ceiling max err {r['max_ceiling_error_cm']} cm | footprint err {r['footprint_error_pct']}% | "
              f"overlaps {r['overlaps_detected']}")
        print(f"  CI95 coverage: {r['ci95_coverage_pct']}")
        print(f"  gates: {_fmt_gates(r['gates'])}")

    rp = rep["repeatability"]
    print(f"\n[REPEATABILITY] {rp['room_id']} evidence={rp.get('evidence')}: max wall diff {rp.get('max_wall_diff_cm')} cm "
          f"({rp.get('max_wall_diff_pct')}%), ceiling spread {rp.get('ceiling_spread_cm')} cm, "
          f"bias {rp.get('ceiling_mean_bias_cm')} cm -> {rp.get('classification')} | gate {rp['overall_repeatability_gate']}")

    d = rep["drift_ablation"]
    print(f"\n[DRIFT ABLATION] {d['method']} | loops {d['loop_closures']} anchors {d['manhattan_anchors']} "
          f"max yaw corr {d['max_yaw_correction_deg']} deg")
    for k in ("drift_correction_ON", "drift_correction_OFF"):
        s = d[k]
        print(f"  {k:21s} rooms {s['rooms_matched']} walls {s['walls_scored']} (missing {s['walls_missing']}) "
              f"wall err {s['wall_error_mean_pct']}% | footprint err {s['footprint_error_pct']}% | "
              f"centroid err {s['mean_centroid_err_cm']}/{s['max_centroid_err_cm']} cm | "
              f"openings<=2cm {s['pct_openings_le_2cm']}% | loop gap {s['loop_closure_gap_cm']} cm")
    print(f"  gate: {d['gate']}")

    dm = rep["damage_evaluation"]
    print(f"\n[DAMAGE] evidence={dm['evidence']} classes {dm['classes_detected']} gate {dm['multi_class_gate']}, "
          f"rules {dm['rules_fired']}, {dm['scope_line_items_count']} scope items. {dm['note']}")
    print(f"\n[TIMING] {rep['timing_s']}\n")


if __name__ == "__main__":
    main()
