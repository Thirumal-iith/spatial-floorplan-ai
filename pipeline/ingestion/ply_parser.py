"""
pipeline/ingestion/ply_parser.py
Universal 3D point cloud parser supporting ASCII & Binary PLY, OBJ, and XYZ formats.
Parses real iPhone LiDAR scans (Record3D, Scaniverse, Polycam, Apple RoomPlan).
"""

import os
import struct
import numpy as np
from typing import Tuple, Optional


class PointCloudParser:
    @staticmethod
    def load_point_cloud(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Loads 3D points and optional RGB colors from .ply, .obj, or .xyz file.
        Returns:
            points: (N, 3) float32 array
            colors: (N, 3) uint8 array or None
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".ply":
            return PointCloudParser._load_ply(file_path)
        elif ext == ".obj":
            return PointCloudParser._load_obj(file_path)
        elif ext in (".xyz", ".txt", ".pts"):
            return PointCloudParser._load_xyz(file_path)
        else:
            raise ValueError(f"Unsupported point cloud format: {ext}")

    @staticmethod
    def _load_ply(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Parses both ASCII and Binary Little-Endian PLY files.
        """
        with open(file_path, "rb") as f:
            header_lines = []
            is_binary = False
            vertex_count = 0
            properties = []

            while True:
                line = f.readline().decode("latin1", errors="ignore").strip()
                header_lines.append(line)
                if line.startswith("format binary_little_endian"):
                    is_binary = True
                elif line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("property"):
                    parts = line.split()
                    prop_type, prop_name = parts[1], parts[2]
                    properties.append((prop_name, prop_type))
                elif line == "end_header":
                    break

            if vertex_count == 0:
                return np.zeros((0, 3), dtype=np.float32), None

            prop_names = [p[0] for p in properties]
            x_idx = prop_names.index("x") if "x" in prop_names else 0
            y_idx = prop_names.index("y") if "y" in prop_names else 1
            z_idx = prop_names.index("z") if "z" in prop_names else 2

            has_colors = all(c in prop_names for c in ("red", "green", "blue"))
            r_idx = prop_names.index("red") if has_colors else -1
            g_idx = prop_names.index("green") if has_colors else -1
            b_idx = prop_names.index("blue") if has_colors else -1

            if not is_binary:
                # ASCII PLY
                points = []
                colors = []
                for _ in range(vertex_count):
                    line = f.readline().decode("latin1", errors="ignore").strip()
                    if not line:
                        break
                    parts = line.split()
                    points.append([float(parts[x_idx]), float(parts[y_idx]), float(parts[z_idx])])
                    if has_colors and len(parts) > max(r_idx, g_idx, b_idx):
                        colors.append([int(parts[r_idx]), int(parts[g_idx]), int(parts[b_idx])])

                pts_arr = np.array(points, dtype=np.float32)
                cls_arr = np.array(colors, dtype=np.uint8) if colors else None
                return pts_arr, cls_arr
            else:
                # Binary Little Endian PLY
                type_map = {
                    "float": "f", "float32": "f", "double": "d", "float64": "d",
                    "uchar": "B", "uint8": "B", "char": "b", "int8": "b",
                    "short": "h", "int16": "h", "ushort": "H", "uint16": "H",
                    "int": "i", "int32": "i", "uint": "I", "uint32": "I"
                }
                fmt_str = "<" + "".join([type_map.get(p[1], "f") for p in properties])
                record_size = struct.calcsize(fmt_str)

                data = f.read(vertex_count * record_size)
                num_records = len(data) // record_size
                points = np.zeros((num_records, 3), dtype=np.float32)
                colors = np.zeros((num_records, 3), dtype=np.uint8) if has_colors else None

                for i in range(num_records):
                    unpacked = struct.unpack_from(fmt_str, data, i * record_size)
                    points[i] = [unpacked[x_idx], unpacked[y_idx], unpacked[z_idx]]
                    if has_colors:
                        colors[i] = [int(unpacked[r_idx]), int(unpacked[g_idx]), int(unpacked[b_idx])]

                return points, colors

    @staticmethod
    def _load_obj(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        points = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.strip().split()
                    points.append([float(parts[1]), float(parts[2]), float(parts[3])])
        return np.array(points, dtype=np.float32), None

    @staticmethod
    def _load_xyz(file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        pts = np.loadtxt(file_path, usecols=(0, 1, 2), dtype=np.float32)
        return pts, None

    @staticmethod
    def save_ply_ascii(file_path: str, points: np.ndarray, colors: Optional[np.ndarray] = None):
        """
        Saves points and optional colors to standard ASCII PLY file.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        has_colors = colors is not None and len(colors) == len(points)
        num_pts = len(points)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_pts}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            if has_colors:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            f.write("end_header\n")

            for i in range(num_pts):
                p = points[i]
                if has_colors:
                    c = colors[i]
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
                else:
                    f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
