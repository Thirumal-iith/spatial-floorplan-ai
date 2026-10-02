# Part 4: One-Page Fix Declaration

### 1. The Single Worst-Performing Gate in Own Benchmark
* **Gate Name:** `photo_stitch_overlap_gate` (Part 2 & Page 3: "Photo-tier whole-property stitch: Per-room photo folders produce one stitched plan with correct adjacency and no room overlaps... A photo path that handles single rooms only fails this row").
* **Failing Number:** `Overlaps Detected: True` (1 collision pair detected between `room_living` and `room_kitchen`, overlapping by $3.42\text{ m}^2$).
* **Baseline Gate Status:** **FAIL**.

---

### 2. Root-Cause Hypothesis and Evidence
* **Evidence:** In the initial benchmark run, inspecting the placed polygons revealed that `room_kitchen` was translated across the shared hallway boundary in the opposite direction (+Y instead of -Y), placing its bounding box directly over `room_living`.
* **Root Cause Hypothesis:**
  In `pipeline/geometry/stitching.py`, the portal alignment calculation `_align_room_to_portal` computed the portal normal transformation as:
  $$t_{\text{portal}} = t_{\text{center}} + (t_{\text{normal}} \cdot \text{wall\_thickness})$$
  However, the candidate room's internal coordinate frame has its own local centroid. When two door portals share similar 2D orientation vectors, the simple rotation $((\theta_{\text{target}} + \pi) - \theta_{\text{cand}})$ fails to resolve the parity of the interior vs exterior half-plane. It placed the candidate room's geometry protruding *into* the corridor or adjacent room rather than projecting *outward* into free space. Because the polygon collision detector was purely passive and lacked an active repulsive relaxation step, the overlap persisted into the final output.

---

### 3. The Fix Intended to Ship and Predicted Outcome
* **Intended Fix:**
  1. **Outward Projection Parity Check:** In `pipeline/geometry/stitching.py`, check the candidate polygon's centroid relative to the target portal normal. If the centroid projects onto the same side as the target room's centroid ($\vec{d}_{\text{cand}} \cdot \vec{n}_{\text{target}} < 0$), invert the portal normal by $180^\circ$ to guarantee the new room projects into open exterior space.
  2. **Active Non-Overlap Spatial Relaxation:** Add an automated collision-resolution loop: if any geometric intersection area $> 0.01\text{ m}^2$ is detected between the newly placed room and existing rooms, iteratively test $90^\circ / 180^\circ$ portal attachment rotations and translate along the unconstrained hallway axis until polygon intersection is exactly $0.00\text{ m}^2$.
* **Predicted Number After Fix:**
  * `Overlaps Detected: False (0 collisions)`
  * `Overlap Area: 0.00 m²`
  * **Predicted Gate Status:** **PASS (Full Marks)**.
