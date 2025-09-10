# app.py
import os
import io
import json
import pandas as pd
from flask import Flask, request, render_template, send_file, jsonify
from audio_utils import (
    register_file, resolve_file,
    extract_formants_raw, extract_formants_image,
    compute_bark_energy, BARK_EDGES,
    compute_frequency_distance_matrix, compute_formant_time_ranges,
    compute_raw_distance_matrix, compute_raw_time_ranges,
)

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Multipart with: file, method, and optional params
    OR JSON with: file_id, method, params
    """
    method = None
    params = {}
    path = None
    file_id = None

    if request.content_type and "multipart/form-data" in request.content_type:
        method = request.form.get("method", "raw")
        params_json = request.form.get("params")
        if params_json:
            params = json.loads(params_json)
        f = request.files["file"]
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        file_id = register_file(path)
    else:
        payload = request.get_json(force=True)
        method = payload.get("method", "raw")
        params = payload.get("params", {})
        file_id = payload.get("file_id")
        path = resolve_file(file_id)
        if not path:
            return jsonify({"error": "file_id not found"}), 400

    denoise = bool(params.get("denoise", False))

    if method == "image":
        percentile = float(params.get("percentile", 90))
        max_freq_range = float(params.get("max_freq_range", 500))
        min_len = int(params.get("min_len", 20))
        res = extract_formants_image(path, percentile, max_freq_range, min_len, denoise=denoise)

        dist_df = compute_frequency_distance_matrix(res["contours"])
        time_df = compute_formant_time_ranges(res["contours"])

        S_db = res["S_db"]
        n_fft = (S_db.shape[0] - 1) * 2
        bark_energy = compute_bark_energy(S_db, res["sr"], n_fft)

        return jsonify({
            "file_id": file_id,
            "mode": "Image-Based",
            "denoised": denoise,
            "sr": res["sr"],
            "time": res["tvals"].tolist(),
            "freq": res["fvals"].tolist(),
            "S_db": res["S_db"].tolist(),
            "contours": res["contours"],
            "bark": {
                "edges": BARK_EDGES.tolist(),
                "energy": bark_energy
            },
            "tables": {
                "distance_csv": dist_df.to_csv(index=True),
                "time_csv": time_df.to_csv(index=False)
            }
        })

    else:
        max_formants = int(params.get("max_formants", 20))
        dur_limit_sec = float(params.get("dur_limit_sec", 5.0))
        res = extract_formants_raw(path, max_formants=max_formants, dur_limit_sec=dur_limit_sec, denoise=denoise)

        dist_df = compute_raw_distance_matrix(res["formants"])
        time_df = compute_raw_time_ranges(res["times"], res["formants"])

        S_db = res["S_db"]
        n_fft = (S_db.shape[0] - 1) * 2
        bark_energy = compute_bark_energy(S_db, res["sr"], n_fft)

        tracks = []
        for i, tr in enumerate(res["formants"]):
            tracks.append({"name": f"F{i+1}", "time": res["times"].tolist(), "freq": tr.tolist()})
        f0 = {"name": "F0", "time": res["times"].tolist(), "freq": res["f0"].tolist()}

        return jsonify({
            "file_id": file_id,
            "mode": "Raw Audio",
            "denoised": denoise,
            "sr": res["sr"],
            "time": res["tvals"].tolist(),
            "freq": res["fvals"].tolist(),
            "S_db": res["S_db"].tolist(),
            "f0": f0,
            "formants": tracks,
            "bark": {
                "edges": BARK_EDGES.tolist(),
                "energy": bark_energy
            },
            "tables": {
                "distance_csv": dist_df.to_csv(index=True),
                "time_csv": time_df.to_csv(index=False)
            }
        })

@app.route("/download/csv", methods=["POST"])
def download_csv():
    data = request.get_json(force=True)
    csv_text = data.get("csv", "")
    filename = data.get("filename", "table.csv")
    buf = io.BytesIO(csv_text.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="text/csv")

if __name__ == "__main__":
    app.run(debug=True)

"""
@echo off
cd /d "%~dp0"
start "" cmd /k python app.py
start "" http://localhost:5000/

"""
