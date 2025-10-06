# app.py
import os
import io
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import cv2

from flask import (
    Flask, request, render_template, send_file, jsonify, send_from_directory
)
from werkzeug.utils import secure_filename

# ===== your existing audio/DSP helpers =====
from audio_utils import (
    register_file, resolve_file,
    extract_formants_raw, extract_formants_image,
    compute_bark_energy, BARK_EDGES,
    compute_frequency_distance_matrix, compute_formant_time_ranges,
    compute_raw_distance_matrix, compute_raw_time_ranges,
)

# ================= Flask & paths =================
app = Flask(__name__, static_folder="static", template_folder="templates")

ROOT = Path(__file__).parent.resolve()
UPLOAD_DIR = ROOT / "uploads"
SPEC_DIR = ROOT / "static" / "specs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR.mkdir(parents=True, exist_ok=True)

# restrict uploads
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB

ALLOWED_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}
def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS

# Optional: keep BLAS threads modest
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("MKL_NUM_THREADS","1")


# =============== Utilities ===============
def _clean_array(a, fill=0.0):
    """Replace NaN/±Inf with finite values so JSON is valid."""
    arr = np.asarray(a, dtype=float)
    return np.nan_to_num(arr, nan=fill, posinf=fill, neginf=fill)

# --- ML mask built directly on S_db (visual clean only) ---
# We normalize S_db to 0..255, compute simple features (intensity, local mean/std),
# then use a GMM + XGB pseudo-label approach to predict a foreground mask.

from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier

def _ml_mask_from_Sdb(S_db, open_k=3, close_k=5):
    """
    Build a foreground mask from a dB spectrogram S_db (2D float array).
    Returns a uint8 0/1 mask with the same shape.
    """
    S = np.asarray(S_db, dtype=float)
    # robust min/max (guard NaNs)
    smin = np.nanmin(S) if np.isfinite(np.nanmin(S)) else -80.0
    smax = np.nanmax(S) if np.isfinite(np.nanmax(S)) else 0.0
    if smax <= smin:
        smax = smin + 1e-6
    # normalize to 0..255 uint8 for CV ops
    S_u8 = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # features: intensity, local mean, local std
    img = S_u8.astype(np.float32)
    h, w = img.shape
    intensity = img.reshape(-1, 1)
    local_mean = cv2.blur(img, (3, 3)).reshape(-1, 1)
    sq_diff = (img - local_mean.reshape(h, w)) ** 2
    local_std = np.sqrt(np.maximum(cv2.blur(sq_diff, (3, 3)), 0.0)).reshape(-1, 1)
    feats = np.hstack([intensity, local_mean, local_std])  # (h*w, 3)

    # GMM (2 clusters) -> brighter cluster id
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(feats)
    hi = int(np.argmax(gmm.means_[:, 0]))
    mask_gmm = (labels == hi).astype(np.uint8).reshape(h, w)

    # XGB with pseudo-labels (top-20% intensity considered foreground)
    pseudo = (S_u8.flatten() > np.percentile(S_u8, 80)).astype(int)
    xgb = XGBClassifier(
        n_estimators=50, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss"
    )
    xgb.fit(feats, pseudo)
    pred = xgb.predict(feats).astype(np.uint8).reshape(h, w)

    # Hybrid AND
    mask = cv2.bitwise_and(mask_gmm, pred)

    # Morph cleanup
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_k, open_k), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))
    return mask  # 0/1


# ===============================
# Main page (existing analyzer UI)
# ===============================
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ===============================
# Analyzer API (supports clean_mode)
# ===============================
@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Multipart with: file, method, and optional params
    OR JSON with: file_id, method, params

    params.clean_mode in {'none','specsub','mlmask'} (optional)
    - 'specsub' uses spectral subtraction in audio_utils
    - 'mlmask' visually cleans S_db only (does not change audio)
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

        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "No file provided"}), 400

        if not _allowed(f.filename):
            return jsonify({"error": "Unsupported file type"}), 400

        safe_name = secure_filename(f.filename)
        path = str(UPLOAD_DIR / safe_name)
        f.save(path)
        file_id = register_file(path)
    else:
        payload = request.get_json(force=True, silent=True)
        if not payload:
            return jsonify({"error": "Invalid JSON payload"}), 400

        method = payload.get("method", "raw")
        params = payload.get("params", {})
        file_id = payload.get("file_id")
        path = resolve_file(file_id)
        if not path:
            return jsonify({"error": "file_id not found"}), 400

    # Clean mode normalization (keep legacy 'denoise' too)
    denoise_flag = bool(params.get("denoise", False))
    clean_mode = params.get("clean_mode")
    if clean_mode not in ("none", "specsub", "mlmask"):
        clean_mode = "specsub" if denoise_flag else "none"

    if method == "image":
        percentile = float(params.get("percentile", 90))
        max_freq_range = float(params.get("max_freq_range", 500))
        min_len = int(params.get("min_len", 20))

        res = extract_formants_image(
            path,
            percentile=percentile,
            max_freq_range=max_freq_range,
            min_len=min_len,
            denoise=(clean_mode == "specsub"),
        )

        # sanitize arrays
        S_db = _clean_array(res["S_db"])
        tvals = _clean_array(res["tvals"])
        fvals = _clean_array(res["fvals"])

        # optional ML visual mask on S_db
        if clean_mode == "mlmask":
            mask = _ml_mask_from_Sdb(S_db)
            S_floor = float(np.nanmin(S_db)) if np.isfinite(np.nanmin(S_db)) else -80.0
            S_db = np.where(mask > 0, S_db, S_floor)

        # compute bark energy on the (possibly masked) S_db
        n_fft_actual = (S_db.shape[0] - 1) * 2
        bark_energy = compute_bark_energy(S_db, res["sr"], n_fft_actual)

        dist_df = compute_frequency_distance_matrix(res["contours"])
        time_df = compute_formant_time_ranges(res["contours"])

        return jsonify({
            "file_id": file_id,
            "mode": "Image-Based",
            "denoised": (clean_mode == "specsub"),
            "clean_mode": clean_mode,
            "sr": res["sr"],
            "time": tvals.tolist(),
            "freq": fvals.tolist(),
            "S_db": S_db.tolist(),
            "contours": res["contours"],
            "bark": {"edges": BARK_EDGES.tolist(), "energy": bark_energy},
            "tables": {
                "distance_csv": dist_df.to_csv(index=True),
                "time_csv": time_df.to_csv(index=False),
            },
        })

    else:
        max_formants = int(params.get("max_formants", 20))
        dur_limit_sec = float(params.get("dur_limit_sec", 5.0))

        res = extract_formants_raw(
            path,
            max_formants=max_formants,
            dur_limit_sec=dur_limit_sec,
            denoise=(clean_mode == "specsub"),
        )

        # sanitize arrays
        S_db = _clean_array(res["S_db"])
        tvals = _clean_array(res["tvals"])
        fvals = _clean_array(res["fvals"])
        times = _clean_array(res["times"])
        f0 = _clean_array(res["f0"])
        formants = [_clean_array(tr) for tr in res["formants"]]

        # optional ML visual mask on S_db (does not change F0/formants)
        if clean_mode == "mlmask":
            mask = _ml_mask_from_Sdb(S_db)
            S_floor = float(np.nanmin(S_db)) if np.isfinite(np.nanmin(S_db)) else -80.0
            S_db = np.where(mask > 0, S_db, S_floor)

        # bark
        n_fft_actual = (S_db.shape[0] - 1) * 2
        bark_energy = compute_bark_energy(S_db, res["sr"], n_fft_actual)

        # tables use cleaned arrays (no NaNs)
        dist_df = compute_raw_distance_matrix(formants)
        time_df = compute_raw_time_ranges(times, formants)

        tracks = [
            {"name": f"F{i+1}", "time": times.tolist(), "freq": tr.tolist()}
            for i, tr in enumerate(formants)
        ]
        f0_obj = {"name": "F0", "time": times.tolist(), "freq": f0.tolist()}

        return jsonify({
            "file_id": file_id,
            "mode": "Raw Audio",
            "denoised": (clean_mode == "specsub"),
            "clean_mode": clean_mode,
            "sr": res["sr"],
            "time": tvals.tolist(),
            "freq": fvals.tolist(),
            "S_db": S_db.tolist(),
            "f0": f0_obj,
            "formants": tracks,
            "bark": {"edges": BARK_EDGES.tolist(), "energy": bark_energy},
            "tables": {
                "distance_csv": dist_df.to_csv(index=True),
                "time_csv": time_df.to_csv(index=False),
            },
        })


# ===============================
# CSV download (existing)
# ===============================
@app.route("/download/csv", methods=["POST"])
def download_csv():
    data = request.get_json(force=True)
    csv_text = data.get("csv", "")
    filename = data.get("filename", "table.csv")
    buf = io.BytesIO(csv_text.encode("utf-8"))
    buf.seek(0)
    return send_file(buf, as_attachment=True, download_name=filename, mimetype="text/csv")


# ===============================
# Health check
# ===============================
@app.route("/healthz")
def healthz():
    return "ok", 200


# ===================================================================
# Spectrogram Comparator (separate page + upload API + assets)
# ===================================================================

# ---- NeonGreen LUT (OpenCV) for comparator images ----
def neon_green_colormap():
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        g = min(255, int(2 * i))
        b = int(0.3 * g)
        r = int(0.1 * g)
        lut[i, 0] = (b, g, r)  # BGR order for OpenCV
    return lut

# ---- Simple spectrogram writer with NeonGreen (comparator) ----
def _save_spectrogram_neon(y, sr, save_path: Path):
    D = librosa.stft(y)
    S = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    S = cv2.normalize(S, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    S = cv2.flip(S, 0)
    S_bgr = cv2.cvtColor(S, cv2.COLOR_GRAY2BGR)
    S_neon = cv2.LUT(S_bgr, neon_green_colormap())
    cv2.imwrite(str(save_path), S_neon)

# ---- Comparator ML-ish denoise path (uint8 spec -> mask) ----
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160

def _load_audio_resampled(path, target_sr=SAMPLE_RATE, duration=None):
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    if duration is not None:
        n = int(target_sr * duration)
        if len(y) > n:
            y = y[:n]
        elif len(y) < n:
            y = np.pad(y, (0, n - len(y)), mode="constant")
    return y.astype(np.float32), target_sr

def _wav_to_spec_uint8(path, duration=None):
    y, sr = _load_audio_resampled(path, target_sr=SAMPLE_RATE, duration=duration)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH)) ** 2  # power
    S_log = np.log1p(S)
    S_norm = cv2.normalize(S_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return S_norm

def _extract_features(spectrogram, window_size=3):
    h, w = spectrogram.shape
    img = spectrogram.astype(np.float32)
    intensity = img.reshape(-1, 1)
    local_mean = cv2.blur(img, (window_size, window_size)).reshape(-1, 1)
    sq_diff = (img - local_mean.reshape(h, w)) ** 2
    local_std = np.sqrt(np.maximum(cv2.blur(sq_diff, (window_size, window_size)), 0.0)).reshape(-1, 1)
    return np.hstack([intensity, local_mean, local_std])

def _classify_gmm(features, shape):
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(features)
    hi = int(np.argmax(gmm.means_[:, 0]))
    return (labels == hi).astype(np.uint8).reshape(shape)

def _classify_gbt(features, spectrogram, shape):
    pseudo = (spectrogram.flatten() > np.percentile(spectrogram, 80)).astype(int)
    clf = XGBClassifier(
        n_estimators=50, max_depth=5, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss"
    )
    clf.fit(features, pseudo)
    return clf.predict(features).reshape(shape).astype(np.uint8)

def _classify_hybrid(features, spectrogram, shape):
    return cv2.bitwise_and(_classify_gmm(features, shape),
                           _classify_gbt(features, spectrogram, shape))

def _knn_denoise(spectrogram, method="hybrid"):
    feats = _extract_features(spectrogram)
    if method == "gmm":
        mask = _classify_gmm(feats, spectrogram.shape)
    elif method == "gbt":
        mask = _classify_gbt(feats, spectrogram, spectrogram.shape)
    else:
        mask = _classify_hybrid(feats, spectrogram, spectrogram.shape)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return spectrogram * mask

def _process_file_for_compare(file_path: Path):
    # Original neon spectrogram (native SR)
    y, sr = librosa.load(str(file_path), sr=None, mono=True)
    duration = float(librosa.get_duration(y=y, sr=sr))
    base = file_path.stem + "_" + uuid.uuid4().hex[:6]

    og_path = SPEC_DIR / f"{base}_og.png"
    _save_spectrogram_neon(y, sr, og_path)

    # Cleaned neon spectrogram via mask
    spec = _wav_to_spec_uint8(str(file_path), duration=None)
    spec_clean = _knn_denoise(spec, method="hybrid")
    img = spec_clean - spec_clean.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype("uint8")
    img = cv2.flip(img, 0)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_color = cv2.LUT(img_bgr, neon_green_colormap())
    clean_path = SPEC_DIR / f"{base}_clean.png"
    cv2.imwrite(str(clean_path), img_color)

    return {
        "id": base,
        "og_url": f"/static/specs/{og_path.name}",
        "clean_url": f"/static/specs/{clean_path.name}",
        "duration": duration,
        "audio_url": f"/uploads/{file_path.name}",
    }

# ---- serve uploaded audio used by comparator player ----
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

# ---- comparator page ----
@app.route("/compare", methods=["GET"])
def compare_index():
    return render_template("compare.html")

# ---- comparator upload API ----
@app.route("/compare/upload", methods=["POST"])
def compare_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    filename = secure_filename(f.filename)
    if not _allowed(filename):
        return jsonify({"error": "Bad extension"}), 400
    save_path = UPLOAD_DIR / filename
    f.save(save_path)
    try:
        return jsonify(_process_file_for_compare(save_path))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===============================
# Entrypoint
# ===============================
if __name__ == "__main__":
    app.run(debug=True)

"""
@echo off
cd /d "%~dp0"
start "" cmd /k python app.py
start "" http://localhost:5000/

"""
