"""
scripts/reproduce_all.py
Master reproduction bundle script.
Executes the end-to-end pipeline, regenerates all benchmark figures,
evaluates all 5 gates across all 3 tiers, runs the Part 3 Head-to-Head test,
and runs the Part 4 Fix Loop before/after validation in under 10 seconds.
"""

import sys
import os
import subprocess
import time


def run_command(cmd, desc):
    print(f"\n>>> [REPRODUCTION] {desc}...")
    start = time.time()
    res = subprocess.run([sys.executable] + cmd,
                         capture_output=True, text=True)
    dur = round(time.time() - start, 2)
    if res.returncode != 0:
        print(f"FAILED (code {res.returncode}):\n{res.stderr}")
        return False
    print(res.stdout.strip())
    print(f">>> Finished in {dur}s")
    return True


def main():
    print("==================================================================")
    print("   MASTER REPRODUCTION BUNDLE: SPATIAL AI CASE STUDY AUDIT        ")
    print("==================================================================")

    # 1. Synthesize benchmark dataset & ground truth
    if not run_command(["-m", "benchmark.dataset_generator"], "Generating ground truth & multi-tier dataset"):
        sys.exit(1)

    # 2. Run Single CLI Pipeline Test
    if not run_command(["-m", "pipeline.run", "--input", "benchmark/data/tier3_lidar_raw", "--tier", "lidar", "--output", "results"], "Testing single-command pipeline on LiDAR tier"):
        sys.exit(1)

    # 3. Run Benchmark Gates Evaluation
    if not run_command(["-m", "benchmark.evaluate"], "Evaluating precision gates across all 3 tiers & drift ablation"):
        sys.exit(1)

    # 4. Run Part 3 Head-to-Head against Magicplan
    if not run_command(["-m", "benchmark.head_to_head"], "Evaluating Part 3 Head-to-Head vs Magicplan"):
        sys.exit(1)

    # 5. Run Part 4 Fix Loop Before and After runs
    if not run_command(["-m", "fix_loop.run_before"], "Running Part 4 Fix Loop BEFORE run (failing gate)"):
        sys.exit(1)

    if not run_command(["-m", "fix_loop.run_after"], "Running Part 4 Fix Loop AFTER run (passing gate)"):
        sys.exit(1)

    print("\n==================================================================")
    print("   ALL ARTIFACTS AND BENCHMARK GATES REPRODUCED SUCCESSFULLY!     ")
    print("==================================================================")
    print("Outputs available in:")
    print("  - Results & Floor Plan: results/floor_plan.svg, results/plan.json")
    print("  - Benchmark Gates:      benchmark/reports/benchmark_results.json")
    print("  - Head-to-Head:         benchmark/head_to_head/head_to_head_report.md")
    print("  - Fix Loop Bundle:      fix_loop/declaration.md, fix_loop/after_run/after_results.json")
    print("  - Compliance Matrix:    reports/compliance_matrix.md")
    print("  - Technical Report:     reports/technical_report.md\n")


if __name__ == "__main__":
    main()
