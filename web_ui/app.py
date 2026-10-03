"""
web_ui/app.py
Interactive Inspection & Spatial Defense Dashboard.
Provides a modern web UI for inspecting floor plans, damage overlays,
concealed-damage rule rationale, insurance restoration line items, and live benchmark gates.
"""

import os
import sys
import json

# Ensure project root is in sys.path regardless of execution directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, render_template, jsonify, request, send_file
from pipeline.run import PipelineRunner

app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/plan")
def get_plan():
    tier = request.args.get("tier", "lidar")
    plan_path = os.path.join(PROJECT_ROOT, "results", "plan.json")
    if os.path.exists(plan_path):
        with open(plan_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Plan not found. Run pipeline first."}), 404


@app.route("/api/plan/svg")
def get_plan_svg():
    svg_path = os.path.join(PROJECT_ROOT, "results", "floor_plan.svg")
    if os.path.exists(svg_path):
        return send_file(svg_path, mimetype="image/svg+xml")
    return "SVG not found", 404


@app.route("/api/download-report")
def download_report():
    pdf_path = os.path.join(PROJECT_ROOT, "reports", "spatial_ai_engineering_report.pdf")
    if os.path.exists(pdf_path):
        return send_file(pdf_path, mimetype="application/pdf", as_attachment=False)
    return "PDF Report not found", 404


@app.route("/api/benchmark")
def get_benchmark():
    bench_path = os.path.join(PROJECT_ROOT, "benchmark", "reports", "benchmark_results.json")
    if os.path.exists(bench_path):
        with open(bench_path, "r") as f:
            data = json.load(f)
        return jsonify(data)
    return jsonify({"error": "Benchmark results not found"}), 404


@app.route("/api/head-to-head")
def get_head_to_head():
    h2h_path = os.path.join(PROJECT_ROOT, "benchmark", "head_to_head", "head_to_head_report.md")
    if os.path.exists(h2h_path):
        with open(h2h_path, "r") as f:
            content = f.read()
        return jsonify({"markdown": content})
    return jsonify({"error": "Head to head report not found"}), 404


@app.route("/api/run-pipeline", methods=["POST"])
def run_pipeline_api():
    req = request.get_json() or {}
    tier = req.get("tier", "lidar")
    input_folder = os.path.join(PROJECT_ROOT, "benchmark", "data", f"tier{3 if tier=='lidar' else (2 if tier=='video' else 1)}_{tier}")
    runner = PipelineRunner(drift_correction=True)
    plan = runner.process_capture(input_folder, tier=tier)
    
    # Save results
    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)
    
    runner.renderer.render_svg(plan, os.path.join(results_dir, "floor_plan.svg"))
    return jsonify(plan)


@app.route("/api/load-sample", methods=["POST"])
def load_sample_api():
    req = request.get_json() or {}
    sample_key = req.get("sample", "lidar_living")

    samples_map = {
        "lidar_living": (os.path.join(PROJECT_ROOT, "sample_captures", "living_room_lidar.ply"), "lidar"),
        "lidar_bedroom": (os.path.join(PROJECT_ROOT, "sample_captures", "bedroom_lidar.ply"), "lidar"),
        "video_walkthrough": (os.path.join(PROJECT_ROOT, "sample_captures", "room_walkthrough.mp4"), "video"),
        "photo_damage": (os.path.join(PROJECT_ROOT, "sample_captures", "water_damage_wall.jpg"), "photos"),
        "multi_room": (os.path.join(PROJECT_ROOT, "benchmark", "data", "tier3_lidar"), "lidar"),
    }

    if sample_key not in samples_map:
        return jsonify({"error": f"Unknown sample: {sample_key}"}), 400

    target_path, tier = samples_map[sample_key]
    runner = PipelineRunner(drift_correction=True)
    plan = runner.process_capture(target_path, tier=tier)

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)

    runner.renderer.render_svg(plan, os.path.join(results_dir, "floor_plan.svg"))
    return jsonify(plan)


@app.route("/api/upload-file", methods=["POST"])
def upload_file_api():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    tier = request.form.get("tier", "lidar")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
        tier = "video"
    elif ext in (".jpg", ".jpeg", ".png", ".heic", ".bmp"):
        tier = "photos"
    elif ext in (".ply", ".obj", ".xyz", ".pts"):
        tier = "lidar"

    upload_dir = os.path.join(PROJECT_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    save_path = os.path.join(upload_dir, file.filename)
    file.save(save_path)

    runner = PipelineRunner(drift_correction=True)
    try:
        plan = runner.process_capture(save_path, tier=tier)
        results_dir = os.path.join(PROJECT_ROOT, "results")
        os.makedirs(results_dir, exist_ok=True)
        with open(os.path.join(results_dir, "plan.json"), "w") as f:
            json.dump(plan, f, indent=2)

        runner.renderer.render_svg(plan, os.path.join(results_dir, "floor_plan.svg"))
        return jsonify(plan)
    except Exception as e:
        return jsonify({"error": f"Processing error: {str(e)}"}), 500


@app.route("/api/custom-sandbox", methods=["POST"])
def custom_sandbox_api():
    """
    Live spatial sandbox: user sets custom room dimensions and damage parameters,
    and the pipeline computes real 3D geometry, plane RANSAC, concealed damage flags, and scoping lines.
    """
    from sample_captures.generate_samples import generate_sample_ply_room

    data = request.get_json() or {}
    w = float(data.get("width_m", 5.0))
    l = float(data.get("length_m", 4.0))
    h = float(data.get("height_m", 2.7))
    water_h = float(data.get("water_height_aff", 0.65))
    has_damage = bool(data.get("has_damage", True))

    sandbox_ply = os.path.join(PROJECT_ROOT, "uploads", "sandbox_custom.ply")
    os.makedirs(os.path.dirname(sandbox_ply), exist_ok=True)

    generate_sample_ply_room(sandbox_ply, width_m=w, length_m=l, height_m=h, has_door=True, has_damage=has_damage)

    runner = PipelineRunner(drift_correction=True)
    plan = runner.process_capture(sandbox_ply, tier="lidar")

    results_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=2)

    runner.renderer.render_svg(plan, os.path.join(results_dir, "floor_plan.svg"))
    return jsonify(plan)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

