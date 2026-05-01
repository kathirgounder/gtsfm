#!/usr/bin/env python3
"""Lightweight Flask server for the GTSFM pipeline trace visualizer.

Sibling to ./viz — completely separate code path. Scans benchmark_results/ (or
custom --base) for pipeline_trace/manifest.json files, serves the manifest +
per-stage COLMAP files to a Babylon.js frontend that animates the pipeline.

Run: ./pipeline-viz [--base benchmark_results] [--port 5174]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from flask import Flask, abort, jsonify, render_template, send_file
from werkzeug.utils import safe_join

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


BASE_DIR = Path(os.environ.get("PIPELINE_VIZ_BASE", "benchmark_results")).resolve()


def find_pipeline_runs(base_dir: Path) -> list[dict]:
    """Find all pipeline traces under base_dir.

    A pipeline trace is any directory containing `manifest.json` AND at least one
    `stage_*/` subfolder with COLMAP files. Returns one entry per trace.
    """
    runs = []
    for manifest_path in base_dir.rglob("manifest.json"):
        trace_dir = manifest_path.parent
        # Must look like a pipeline trace dir (has stage_* subfolders)
        if not any(trace_dir.glob("stage_*")):
            continue
        rel = trace_dir.relative_to(base_dir).as_posix()
        # Friendly label: drop trailing "pipeline_trace" segment for readability
        parts = rel.split("/")
        label = "/".join([p for p in parts if p != "pipeline_trace"]) or rel
        try:
            with manifest_path.open() as f:
                manifest = json.load(f)
            num_stages = len(manifest.get("stages", []))
        except Exception:
            num_stages = 0
        runs.append({
            "id": rel,
            "label": label,
            "num_stages": num_stages,
            "manifest_url": f"/api/runs/{quote(rel)}/manifest",
            "data_url_prefix": f"/data/{quote(rel)}",
        })
    runs.sort(key=lambda r: r["id"])
    return runs


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/runs")
def list_runs():
    runs = find_pipeline_runs(BASE_DIR)
    return jsonify({"base_dir": str(BASE_DIR), "count": len(runs), "items": runs})


@app.get("/api/runs/<path:run_id>/manifest")
def get_manifest(run_id):
    abs_path = safe_join(str(BASE_DIR), run_id, "manifest.json")
    if abs_path is None:
        abort(404)
    p = Path(abs_path).resolve()
    try:
        p.relative_to(BASE_DIR)
    except ValueError:
        abort(403)
    if not p.exists():
        abort(404)
    with p.open() as f:
        return jsonify(json.load(f))


@app.get("/data/<path:subpath>")
def serve_data(subpath):
    abs_path = safe_join(str(BASE_DIR), subpath)
    if abs_path is None:
        abort(404)
    p = Path(abs_path).resolve()
    try:
        p.relative_to(BASE_DIR)
    except ValueError:
        abort(403)
    if not p.exists() or not p.is_file():
        abort(404)
    return send_file(str(p), as_attachment=False)


def main():
    parser = argparse.ArgumentParser(description="GTSFM pipeline-viz server", add_help=False)
    parser.add_argument("--base", "-b", default="benchmark_results",
                        help="Base folder to scan for pipeline_trace dirs (default: benchmark_results)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", "-p", type=int, default=5174,
                        help="Port (default: 5174 — sibling to ./viz on 5173).")
    parser.add_argument("--help", action="help", default=argparse.SUPPRESS)
    args = parser.parse_args()

    global BASE_DIR
    BASE_DIR = Path(args.base).resolve()
    print(f"[pipeline-viz] Serving traces from: {BASE_DIR}")
    if not BASE_DIR.exists():
        print(f"[pipeline-viz] WARNING: base dir does not exist; create {BASE_DIR}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
