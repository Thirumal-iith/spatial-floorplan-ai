# Spatial Capture → Floor Plan → Damage Scope Pipeline
Applied AI Engineer Case Study (Aug 2026)

> **Status:** in progress. See [progress.md](progress.md) for the live, honest status of every requirement.
> The current benchmark data under `benchmark/data/` is **synthetic** and is labelled as such in every output
> (`data_provenance` field). Real captures and laser ground truth are being collected.

---

## 1. What the case study asks for

From phone sensors onward, build a system that turns a handheld capture of a property into:

| Output (per capture) | Where it lives |
| :-- | :-- |
| Dimensioned per-room plan: walls, ceiling height, floor area, openings | `plan.json → rooms[]` |
| Stitched multi-room plan with correct adjacency | `plan.json → rooms[].polygon`, `floor_plan.svg` |
| Per-surface damage regions with class and metric extent | `rooms[].walls[].damage_regions[]` |
| Concealed-damage flags with the rule that fired | `concealed_damage_flags[]` |
| Scope line items keyed to surfaces | `scope_line_items[]` |
| A 95% confidence interval on every measurement | every `{val, ci_95}` object |
| One command per capture, JSON to a published schema, rendered plan | `python -m pipeline.run ...` |

**Three input tiers, all mandatory, same output contract:**
1. **Photos:** 2–8 stills per room, one folder per room, no depth or poses. Must still produce a stitched whole-property plan.
2. **Video:** handheld walkthrough clip, iPhone 15 or newer.
3. **LiDAR:** depth + poses + intrinsics on Pro devices.

Intervals must **widen honestly** as sensor data thins. Confident garbage on thin input caps the score.

### Gates
| Gate | Threshold |
| :-- | :-- |
| Opening widths | ≤ 2 cm on ≥ 85% of openings; missed and phantom openings both count as misses |
| Ceiling height | ≤ 1.5 cm per room; spread across repeat captures ≤ 1 cm; report says *biased* vs *unrepeatable* |
| Repeatability | two captures of the same room agree within 1 cm or 0.5% per wall |
| Drift accountability | state the method; ablation of stitched footprint ON vs OFF. "Poses as-is" fails |
| Photo whole-property stitch | correct adjacency, no overlaps, footprint ±8% |
| Tier accuracy | photo walls ±8%, video walls ±3%, calibration scored at every tier |
| Head-to-head (Part 3) | beat or tie a consumer app (Magicplan/Polycam) on ≥ 70% of shared dimensions, 2 rooms |
| Fix loop (Part 4, 25%) | declare worst gate → root cause → predicted number → ship fix → regenerable before/after + diff |
| Process (Part 5) | incremental git history |

### Benchmark set we must build ourselves
- One multi-room capture: ≥ 3 rooms + a connector
- One furnished room with staged damage covering 2 damage classes
- The same rooms at all three tiers (photos as per-room folders)
- At least one room captured twice at the same tier
- Laser/tape ground truth on everything; raw sensor data submitted

### Scoring
30% walk-in test (cold run on the examiners' capture) · 25% fix loop · 15% verified benchmark · 10% compliance matrix · 10% head-to-head · 5% capture route · 5% process evidence.

### Constraints
Handheld consumer capture only. Any pretrained model/dataset/API **with disclosure**. Runs fully offline (no calls to our infrastructure). Weights fetched by script. Must cover mirrors, glass, wet-look surfaces and low light.

---

## 2. Quick start (clean machine, < 15 min)

Requires Python 3.10+ (tested on 3.14, Windows 11).

```bash
pip install -r requirements.txt

# Reproduce every reported number (currently on the synthetic benchmark)
python scripts/reproduce_all.py
```

### One command per capture
```bash
python -m pipeline.run --input <capture_folder_or_file> --tier lidar  --output ./results
python -m pipeline.run --input <capture_folder_or_file> --tier video  --output ./results
python -m pipeline.run --input <folder_with_one_subfolder_per_room> --tier photos --output ./results

# Drift ablation (poses as-is)
python -m pipeline.run --input <capture> --tier lidar --no-drift-correction --output ./results_off
```

Outputs: `results/plan.json` (contract, schema in [pipeline/models.py](pipeline/models.py)) and `results/floor_plan.svg`.

Optional local viewer: `python web_ui/app.py` → http://127.0.0.1:5000

How to capture: [app/capture_protocol.md](app/capture_protocol.md). Hardware support: [app/device_matrix.json](app/device_matrix.json).

---

## 3. Architecture

```
capture (photos | video | LiDAR export)
        │
        ▼
pipeline/ingestion      ply_parser · image_cv · video_cv · room_classifier
        │
        ▼
pipeline/geometry       plane_detection (RANSAC, Manhattan snap)
                        openings (void / jamb detection on wall planes)
                        drift_correction (loop closure + arc-length pose relaxation)
                        photo_reconstruction (single-view layout, scale anchor)
                        stitching (portal-keyed placement + overlap rejection)
        │
        ▼
pipeline/damage         detector (HSV/Lab + edge segmentation → metric extent on wall (u,v))
                        concealed_engine (deterministic rules citing IICRC S500 / NEC / ASTM / EPA)
                        scoping (Xactimate-style line items keyed to room/wall)
        │
        ▼
pipeline/calibration    uncertainty (tier-dependent 95% CIs)
        │
        ▼
pipeline/rendering      plan_renderer → SVG     +    plan.json (pydantic schema)
```

| Folder | Purpose |
| :-- | :-- |
| [pipeline/](pipeline/) | The product: one CLI, all tiers |
| [benchmark/](benchmark/) | Dataset, ground truth, gate evaluation, head-to-head |
| [fix_loop/](fix_loop/) | Part 4 declaration and before/after runs |
| [app/](app/) | Capture protocol (Route 2) and device matrix |
| [reports/](reports/) | Compliance matrix, benchmark report, technical report |
| [scripts/](scripts/) | `reproduce_all.py` and helper scripts |
| [web_ui/](web_ui/) | Optional Flask viewer (not required by the case study) |
| `android_build/`, `android_tools/` | Experimental; not part of the graded capture route (case study targets iPhone) |

---

## 4. Tech stack and why

| Choice | Why | Alternatives rejected |
| :-- | :-- | :-- |
| **Python 3.10+** | Fast iteration on geometry/CV, every examiner machine can run it | C++ (slow to iterate), Swift-only (would lock pipeline to Mac) |
| **NumPy** | RANSAC, plane fitting, pose math are all small dense linear algebra | Open3D (heavy binary; may adopt for ICP if needed) |
| **OpenCV (headless)** | Video decoding, feature tracking, edges/morphology for damage; no GUI deps | scikit-image (slower, fewer video tools) |
| **Pillow** | EXIF (focal length) and HEIC/JPEG loading for photo tier | — |
| **Pydantic v2** | The "published schema": typed contract, validation, JSON Schema export | hand-written dicts (no validation) |
| **Flask** | Tiny optional local viewer, zero cloud | Streamlit (heavier) |
| **SVG output** | Vector plan, scalable, viewable in any browser, diffable | PNG (not dimension-accurate) |
| **Route 2 stock capture** | Install in minutes from the App Store; no TestFlight provisioning risk on the examiners' phone | Route 1 custom ARKit app (needs Mac + Apple dev account) |
| **Deterministic rule engine** | Concealed-damage flags must cite the rule that fired; rules are auditable and defendable live | LLM reasoning (non-deterministic, not reproducible) |
| **No cloud APIs** | Constraint: must run without our infrastructure | — |

Pretrained models: **none used yet**. Any model adopted for the video/photo tiers will be listed here with its licence, and a fetch script will download it.

---

## 5. Key technical decisions

- **Uncertainty:** each measurement is reported as `{val, ci_95}`. Interval width depends on the tier (LiDAR < video < photos) and will be re-fitted from the real residuals on the benchmark, so that ≥ 95% of laser values fall inside.
- **Drift:** loop closure at the start/end of the walk, residual spread along arc length (`drift_correction.py`). The ablation is computed honestly. It currently **fails**: corrected poses are not yet used for room placement (tracked in progress.md).
- **Photo stitching:** rooms are placed by matching door portals (`connected_room_id`) and rejecting overlapping placements.
- **Damage:** classical colour/edge segmentation projected to metric wall coordinates. Rules map damage geometry to building-code-cited concealed flags, then to line items keyed to the surface.
- **Real-world hazards** (mirrors, glass, wet-look floors, low light): mitigations are described in the technical report. Test captures are pending.

---

## 6. Honesty notes

- `benchmark/data/*` is generated by `benchmark/dataset_generator.py`. Its "ground truth" is a design spec, not a laser measurement. Gate results on it only prove that the code runs.
- An earlier fabricated Magicplan export was removed (kept for audit in `scratch/quarantined_fabricated/`). The head-to-head now reports `NOT_RUN` until a real export exists at `benchmark/head_to_head/incumbent_export.json`.
- Report markdown files in `reports/` predate this audit and still contain unverified numbers. They will be regenerated.
