# Progress Tracker

Legend: ✅ real & verified · 🟡 works on synthetic data only · 🟠 code exists but is a passthrough/stub · ❌ missing · 🔒 blocked on real-world capture

Last baseline run: `python scripts/reproduce_all.py` (Oct 3 2026, honest mode).

## 0. Audit findings (what the earlier AI-generated code actually did)

| Finding | Evidence | Action |
| :-- | :-- | :-- |
| Benchmark "captures" are `scenario.json` files containing the ground-truth polygons; the pipeline reads geometry straight from them | [pipeline/run.py](pipeline/run.py) `scenario_file` branch; [benchmark/dataset_generator.py](benchmark/dataset_generator.py) | Now tagged `data_provenance` in every plan.json. Replace with real captures (Phase 3) |
| Photo "captures" are text files named `.jpg` (`JPEG_PLACEHOLDER_...`) | dataset_generator.py | Replace with real photos |
| "Leica DISTO" ground truth was never measured | dataset_generator.py | Re-measure with laser/tape |
| Magicplan export was fabricated in code | old `benchmark/head_to_head.py` | ✅ Removed; quarantined in `scratch/quarantined_fabricated/`. New script needs a real export, otherwise reports NOT_RUN |
| Drift ablation added a constant `+2.85 m²` and hardcoded `38.6 cm / 0.4 cm` | old `benchmark/evaluate.py` | ✅ Removed; now computed. Gate honestly **FAILS** |
| Repeatability 0.0 cm because both "captures" are the same GT file | dataset_generator.py | Needs two real captures |
| Photo tier ceiling = `GT × 0.96` ("synthetic noise") | run.py | Replace with perceived estimate |
| No git repository / `git` not installed | `git log` fails | Install git, `git init`, commit per step |
| Docs link to another machine (`c:/Users/medik/...`) | README, compliance matrix | Rewrite with relative links |

## 1. Requirement status

| ID | Requirement (case study) | Status | Next step |
| :-- | :-- | :-: | :-- |
| P1-a | Capture route (Route 2 one-page protocol) | 🟠 | Rewrite: name exact apps & export settings |
| P1-b | Tier 1 Photos: 2–8 stills/room, folder per room, stitched plan | 🟠 | Real layout estimation + scale anchor |
| P1-c | Tier 2 Video walkthrough | 🟠 | SfM / monocular depth + scale |
| P1-d | Tier 3 LiDAR: depth + poses + intrinsics | � | ✅ 3D Scanner App "All Data" loader + perception (`lidar_capture.py`, `lidar_layout.py`); validated on synthetic renders only. Needs real iPhone capture |
| P1-e | Device matrix with honest accuracy | 🟠 | Fill from measured numbers |
| P2-a | Per-room plan: walls, ceiling, floor area, openings | 🟡 | Perceive from real data |
| P2-b | Stitched multi-room plan, correct adjacency | 🟡 | Pose-driven placement |
| P2-c | Damage regions (class + metric extent) | 🟡 | Run detector on real images, drop GT notes |
| P2-d | Concealed-damage flags with rule | ✅ | Rule engine is deterministic and fine |
| P2-e | Scope line items keyed to surfaces | ✅ | — |
| P2-f | CI on every measurement | 🟡 | Calibrate widths from real residuals |
| P2-g | One command per capture, JSON schema, rendered plan | ✅ | — |
| G1 | Openings ≤2 cm on ≥85%, misses/phantoms scored | 🟡 PASS (synthetic, perceived) | LiDAR: 9/9 matched, 0 phantom, 100% ≤2 cm. Phantoms now scored |
| G2 | Ceiling ≤1.5 cm; spread ≤1 cm; bias vs unrepeatable classified | 🟡 PASS (synthetic) | Ceiling err 0.0 cm, spread 0.0 cm, classified "repeatable and unbiased". Synthetic ceilings are flat, so this is optimistic |
| G3 | Repeatability ≤1 cm or 0.5% per wall | 🟡 PASS (synthetic, perceived) | Max 1.3 cm = 0.317% (passes via 0.5% clause). Fixed false loop closure (radius 1.0→0.5 m) |
| G4 | Drift accountability + ON/OFF footprint ablation | 🟡 PASS (synthetic) | Real SE(2) pose graph: Manhattan yaw anchors + hub loop closures. ON: room centroid err 3.1/5.1 cm mean/max, yaw 0.004°, envelope 1.0%. OFF: 19.3/49.1 cm, 7.15°, envelope 3.5%. Needs a real multi-room capture |
| G5 | Photo stitch: adjacency, no overlaps, footprint ±8% | 🟡 | Real photos |
| G6 | Photo walls ±8%, video ±3%, calibration scored at every tier | 🟡 | CI coverage scored per tier. LiDAR wall coverage was 87.5% → fixed noise model (abs + rel instead of max) → 100%. Video/photo are still passthrough |
| P3 | Head-to-head vs consumer app, ≥70% beat/tie | 🔒 NOT_RUN | Scan 2 rooms with Magicplan/Polycam free tier |
| P4 | Fix loop: declaration, before/after, diff | 🟠 | Redo on the real worst gate |
| P5 | Process evidence (git history) | ❌ | Install git, commit incrementally |
| D1 | Compliance matrix | 🟠 | Rewrite honestly |
| D3 | README → fresh capture < 15 min | 🟠 | ✅ README rewritten; verify on clean machine |
| D4 | Reproduction bundle | 🟡 | Runs; must use real raw data |
| D5 | Benchmark report | 🟠 | Regenerate from results JSON (no hand-typed numbers) |
| D7 | Technical report ≤6 pages | 🟠 | Rewrite after real numbers |
| D8 | Raw benchmark data | 🔒 | Collect |
| C1 | Mirrors, glass, wet-look, low light covered | 🟠 | Implement filters + test captures |
| C2 | Weights fetched by script | ❌ | `scripts/fetch_models.py` if a model is adopted |

## 2. Done
- [x] Read case study and audited codebase
- [x] Removed fabricated Magicplan export and drift constants
- [x] Added `data_provenance` and `drift_metrics` to output contract
- [x] Added `opencv-python-headless` to requirements (was imported but undeclared)
- [x] Reproduction script runs end to end on Python 3.14
- [x] Detailed README
- [x] Drift: replaced fake PGO with a 2D pose graph (yaw LS with Manhattan anchors, then xy LS with loop closures to the start marker); rooms are placed from (corrected or raw) poses
- [x] Synthetic generator: seeded RNG (old `hash()` noise changed every run), looped protocol trajectory, simulated biased odometry, rooms stored in camera frame, GT world placement, GT labelled SYNTHETIC
- [x] Removed ×1.08 footprint fudge; footprint = net area − exact rasterised overlap; overlap via polygons (not bounding boxes)
- [x] Repeatability is now non-trivial (0.6 cm / 0.15% from independent noise), no longer 0.0
- [x] `benchmark/synth_scene.py`: single synthetic scene + ray-cast depth renderer writing the 3D Scanner App layout (233-frame property walk, 126-frame bedroom repeat)
- [x] LiDAR perception from raw depth: Manhattan yaw, pose graph, fused cloud, room segmentation, wall-face snapping, per-room floor/ceiling, openings from wall-face voids with plausibility filter
- [x] Evaluator rewritten: geometric matching, phantom openings, CI coverage per tier, evidence label (perceived/passthrough), repeatability classification, drift ablation on perceived data
- [x] Calibration fix: wall CI = abs + rel·L (independent sources add)
- [x] `reproduce_all.py` and web UI point to `tier3_lidar_raw`; full reproduction runs clean

## 3. Next (in order)
1. Install git, `git init`, first commit of the audited state
2. ~~Drift~~ done. ~~LiDAR ingestion~~ done. ~~Phantom scoring~~ done
3. Update `app/capture_protocol.md`: 3D Scanner App → LiDAR mode → export "All Data"; start on taped X in connector, walk back over it (within 0.5 m) between rooms, finish on it
4. Video tier perception (currently passthrough)
5. Photo tier perception (currently passthrough)
6. Damage detector on rendered/real images (currently passthrough)
7. Mirrors/glass/wet-look/low-light: confidence-map + depth-consistency filters
8. Real capture session (rooms, damage, repeat, laser GT, incumbent app)
9. Video + photo tiers on real data
10. Fix loop on real worst gate
11. Regenerate reports from JSON; rewrite compliance matrix & technical report
