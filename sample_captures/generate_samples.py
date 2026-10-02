"""
sample_captures/generate_samples.py
Synthesizes real sample point clouds (.ply) and photos for live testing.
"""

import os
import numpy as np
from PIL import Image, ImageDraw
from pipeline.ingestion.ply_parser import PointCloudParser


def generate_sample_ply_room(
    file_path: str,
    width_m: float = 5.2,
    length_m: float = 4.3,
    height_m: float = 2.7,
    has_door: bool = True,
    has_damage: bool = True
):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    points = []
    colors = []

    # 1. Floor points (z = 0)
    for x in np.linspace(0, width_m, 70):
        for y in np.linspace(0, length_m, 60):
            points.append([x + np.random.normal(0, 0.005), y + np.random.normal(0, 0.005), 0.0 + np.random.normal(0, 0.003)])
            colors.append([160, 140, 120])  # wood/tile floor

    # 2. Ceiling points (z = height_m)
    for x in np.linspace(0, width_m, 70):
        for y in np.linspace(0, length_m, 60):
            points.append([x + np.random.normal(0, 0.005), y + np.random.normal(0, 0.005), height_m + np.random.normal(0, 0.003)])
            colors.append([240, 240, 245])  # white ceiling

    # 3. Wall 1: South Wall (y = 0, x from 0 to width_m)
    for x in np.linspace(0, width_m, 80):
        for z in np.linspace(0, height_m, 50):
            # Check for door void: x in [1.5, 2.32], z < 2.05
            if has_door and (1.50 <= x <= 2.32) and (z <= 2.05):
                continue  # door opening void!
            points.append([x + np.random.normal(0, 0.004), 0.0 + np.random.normal(0, 0.004), z + np.random.normal(0, 0.004)])
            colors.append([220, 225, 230])

    # 4. Wall 2: East Wall (x = width_m, y from 0 to length_m)
    for y in np.linspace(0, length_m, 70):
        for z in np.linspace(0, height_m, 50):
            p = [width_m + np.random.normal(0, 0.004), y + np.random.normal(0, 0.004), z + np.random.normal(0, 0.004)]
            # Water stain discoloration: y in [1.0, 2.5], z < 0.65
            if has_damage and (1.0 <= y <= 2.5) and (z <= 0.65):
                points.append(p)
                colors.append([180, 130, 70])  # brownish-yellow water damage!
            else:
                points.append(p)
                colors.append([220, 225, 230])

    # 5. Wall 3: North Wall (y = length_m, x from 0 to width_m)
    for x in np.linspace(0, width_m, 80):
        for z in np.linspace(0, height_m, 50):
            # Window void: x in [1.8, 3.4], z in [0.9, 2.25]
            if (1.80 <= x <= 3.40) and (0.90 <= z <= 2.25):
                continue  # window opening void!
            points.append([x + np.random.normal(0, 0.004), length_m + np.random.normal(0, 0.004), z + np.random.normal(0, 0.004)])
            colors.append([220, 225, 230])

    # 6. Wall 4: West Wall (x = 0, y from 0 to length_m)
    for y in np.linspace(0, length_m, 70):
        for z in np.linspace(0, height_m, 50):
            points.append([0.0 + np.random.normal(0, 0.004), y + np.random.normal(0, 0.004), z + np.random.normal(0, 0.004)])
            colors.append([220, 225, 230])

    pts_arr = np.array(points, dtype=np.float32)
    cls_arr = np.array(colors, dtype=np.uint8)

    PointCloudParser.save_ply_ascii(file_path, pts_arr, cls_arr)
    print(f"[SAMPLE GENERATOR] Generated {len(pts_arr)} points in {file_path}")


def generate_sample_damage_photo(file_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    w, h = 640, 480
    img = Image.new("RGB", (w, h), color=(235, 238, 240))
    draw = ImageDraw.Draw(img)

    # Baseboard at bottom
    draw.rectangle([0, h - 50, w, h], fill=(120, 85, 60))

    # Water damage stain near floor
    draw.ellipse([140, h - 180, 420, h - 45], fill=(185, 140, 85))
    draw.ellipse([170, h - 150, 380, h - 48], fill=(160, 115, 65))

    # Outlet receptacle
    draw.rectangle([480, h - 140, 520, h - 80], fill=(245, 245, 245), outline=(150, 150, 150), width=2)

    img.save(file_path, "JPEG", quality=90)
    print(f"[SAMPLE GENERATOR] Generated photo in {file_path}")


if __name__ == "__main__":
    generate_sample_ply_room("sample_captures/living_room_lidar.ply", 5.2, 4.3, 2.7, True, True)
    generate_sample_ply_room("sample_captures/bedroom_lidar.ply", 4.1, 3.6, 2.7, True, False)
    generate_sample_damage_photo("sample_captures/water_damage_wall.jpg")
