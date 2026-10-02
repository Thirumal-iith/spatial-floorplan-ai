# One-Page Stock Capture Protocol (Route 2)
**Standard Operating Procedure for Property Scanning & Spatial Inspection**

This one-page protocol is designed for non-engineers (insurance adjusters, field technicians, homeowners) using standard consumer iOS hardware. Following this literal protocol guarantees valid inputs for the reconstruction, damage detection, and repair-scoping pipeline.

---

### 1. What to Install
* **Supported Devices:** iPhone 15 or newer (Photos/Video); iPhone 15 Pro, 15 Pro Max, 16 Pro, 16 Pro Max (LiDAR Tier).
* **For Tier 1 (Photos) & Tier 2 (Video):** Built-in iOS **Camera** App (Standard 0.5x Ultra-Wide and 1.0x Wide lens).
* **For Tier 3 (LiDAR):** Install **Record3D** or **Scaniverse** (Free from the Apple App Store, 0 configuration required, exports standard `.ply`/`.obj` or synchronized depth `.r3d`/`.zip` bundle).

---

### 2. Physical Preparation (What to Avoid)
1. **Lighting:** Turn ON all room lights. Open interior blinds to equalize exposure. Avoid direct blinding sunlight into the lens.
2. **Doors & Connectors:** Prop all interior connecting doors completely open (minimum 90°). Do NOT move doors during capture.
3. **Reflective Surfaces:** Keep camera angled at ~15° downward or upward when passing large wall mirrors to avoid sensor multipath reflection.
4. **Moving Obstacles:** Ensure pets and other persons remain outside the capture path.

---

### 3. How to Walk & How Long

#### Tier 1: Photos (2 to 8 stills per room)
* Stand at room threshold and take:
  1. Four corner shots: Stand near each corner, pointing towards the diagonal opposite corner showing both wall intersections, floor, and ceiling.
  2. One shot looking straight at the primary opening/doorway connecting to the hallway.
  3. Up to three close-up or orthogonal shots of any visible surface damage.
* **Duration:** 1 to 2 minutes per room.

#### Tier 2: Handheld Video Walkthrough
* Hold phone firmly at chest level with two hands, lens oriented horizontally or 45° forward-down.
* Walk at a calm, constant pace of **0.3 to 0.5 meters/second** (approx. 1 footstep per second).
* Follow a **Perimeter Wall-Hugging Loop**:
  1. Start in the primary connector/hallway.
  2. Enter Room 1, pan smoothly along left wall, ceiling edge, back wall, right wall.
  3. Exit back to hallway, proceed to Room 2, repeat.
  4. **CRITICAL LOOP CLOSURE:** Always return to your exact starting point in the hallway and record the initial doorframe for 3 seconds before stopping.
* **Duration:** 3 to 6 minutes for an entire 3-room + connector apartment.

#### Tier 3: LiDAR Capture
* Open Record3D (or Scaniverse in LiDAR / Splat / Mesh mode).
* Walk the same continuous perimeter loop as Video at **0.3 m/s**.
* Ensure the LiDAR sensor "paints" every wall plane, corner junction, door jamb, and ceiling line.
* Complete the loop closure back at the origin.
* **Duration:** 4 to 8 minutes.

---

### 4. How to Hand Files to the Pipeline
Export files directly via AirDrop, Lightning/USB-C cable, or iCloud Drive into your project folder following this exact directory naming:

```
capture_session/
├── metadata.json          # Optional: property address, operator notes
├── tier1_photos/          # Photos tier
│   ├── room_living/       # 2-8 JPEG/HEIC images
│   ├── room_bedroom/
│   ├── room_kitchen/
│   └── connector_hall/
├── tier2_video/           # Video tier
│   └── walkthrough.mov    # Single 1080p/4K 30/60fps video
└── tier3_lidar/           # LiDAR tier
    ├── frames_depth/      # Synchronized 16-bit depth PNGs or .ply
    ├── poses.json         # 4x4 camera-to-world pose matrices
    └── intrinsics.json    # fx, fy, cx, cy camera intrinsics
```

To run the pipeline on this capture, execute:
```bash
python -m pipeline.run --input /path/to/capture_session --tier lidar
```
All dimensions, confidence intervals, damage masks, concealed-damage flags, and repair scope items will be generated in `results/`.
