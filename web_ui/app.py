"""
web_ui/app.py
Interactive Inspection & Spatial Defense Dashboard.
Provides a modern web UI for inspecting floor plans, damage overlays,
concealed-damage rule rationale, insurance restoration line items, and live benchmark gates.
"""

import os
import json
from flask import Flask, render_template, jsonify, request, send_file
from pipeline.run import PipelineRunner

app = Flask(__name__, template_folder="templates", static_folder="static")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


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


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
