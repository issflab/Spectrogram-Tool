# audio_utils.py
import os
import io
import uuid
import base64
import numpy as np
import pandas as pd
import librosa
import librosa.display
import parselmouth
from skimage import measure
import soundfile as sf

# ---------------------------
# Registry for uploaded files
# ---------------------------
FILE_REGISTRY = {}  # file_id -> absolute path

def register_file(path: str) -> str:
    fid = str(uuid.uuid4())
    FILE_REGISTRY[fid] = path
    return fid

def resolve_file(file_id: str) -> str:
    return FILE_REGISTRY.get(file_id)

# ---------------------------
# Core DSP helpers
# ---------------------------
def compute_spectrogram(y, sr, n_fft=2048, hop_length=256, clip_min_db=-80):
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    S_db = np.clip(S_db, clip_min_db, 0)
    time_vals = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freq_vals = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S_db, time_vals, freq_vals, hop_length

def denoise_spectral_subtract(
    y, sr, noise_duration=0.5, n_fft=2048, hop_length=256, oversub=1.0, floor_db=-80.0
):
    """
    Very simple spectral subtraction:
    1) Estimate noise magnitude from the first `noise_duration` seconds,
    2) Subtract it from each frame (with oversub),
    3) Clamp to a floor in dB,
    4) Rebuild with original phase and iSTFT.
    """
    if len(y) < int(noise_duration * sr):
        noise_duration = max(0.1, len(y) / sr)

    Y = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(Y), np.angle(Y)

    n_frames = max(1, int((noise_duration * sr) / hop_length))
    noise_mag = np.median(mag[:, :n_frames], axis=1, keepdims=True)

    clean_mag = np.maximum(mag - oversub * noise_mag, 0.0)

    mag_db = librosa.amplitude_to_db(clean_mag + 1e-10, ref=np.max)
    mag_db = np.clip(mag_db, a_min=floor_db, a_max=0.0)
    clean_mag = librosa.db_to_amplitude(mag_db, ref=1.0) * np.max(clean_mag + 1e-10)

    Y_clean = clean_mag * np.exp(1j * phase)
    y_hat = librosa.istft(Y_clean, hop_length=hop_length, length=len(y))
    return y_hat

# ---------------------------
# RAW (Parselmouth) pipeline
# ---------------------------
def extract_formants_raw(audio_path, max_formants=20, dur_limit_sec=5.0, denoise=False):
    """
    Extracts F0 and formants using Parselmouth (Burg) and computes a spectrogram.
    If denoise=True, denoise is done in-memory; NO temp files are written.
    """
    # 1) Load full audio at native SR
    y_full, sr_full = librosa.load(audio_path, sr=None, mono=True)

    # 2) Optional in-memory denoise
    y_proc = denoise_spectral_subtract(y_full, sr_full) if denoise else y_full

    # 3) Build Parselmouth Sound directly from array (no temp WAV)
    # Parselmouth expects float64; ensure dtype and pass sampling frequency
    snd = parselmouth.Sound(y_proc.astype(np.float64), sampling_frequency=sr_full)

    # 4) Limit analysis window to dur_limit_sec (if shorter, use full)
    total_dur = snd.get_total_duration()
    window_end = dur_limit_sec if dur_limit_sec > 0 else total_dur
    window_end = min(window_end, total_dur)
    if window_end < total_dur:
        snd_win = snd.extract_part(from_time=0.0, to_time=window_end, preserve_times=True)
    else:
        snd_win = snd

    # 5) Parselmouth pitch/formants on the same window
    pitch = snd_win.to_pitch(time_step=0.01)
    formant = snd_win.to_formant_burg(time_step=0.01)

    # 6) Spectrogram on the same window using the processed audio
    y_seg = y_proc[: int(window_end * sr_full)] if window_end > 0 else y_proc
    S_db, tvals, fvals, hop = compute_spectrogram(y_seg, sr_full)

    # 7) Collect tracks
    times = pitch.xs()
    f0_track = []
    formant_tracks = [[] for _ in range(max_formants)]

    for t in times:
        f0 = pitch.get_value_at_time(t)
        f0_track.append(float(f0) if f0 else 0.0)
        for i in range(1, max_formants + 1):
            try:
                f = formant.get_value_at_time(i, t)
                formant_tracks[i - 1].append(float(f) if f is not None else 0.0)
            except Exception:
                formant_tracks[i - 1].append(0.0)

    return {
        "S_db": S_db,
        "sr": sr_full,
        "hop_length": hop,
        "times": np.array(times, dtype=float),
        "f0": np.array(f0_track, dtype=float),
        "formants": [np.array(tr, dtype=float) for tr in formant_tracks],
        "tvals": tvals,
        "fvals": fvals,
    }


# ---------------------------
# IMAGE (Contours) pipeline
# ---------------------------
def extract_formants_image(audio_path, percentile=90, max_freq_range=500, min_len=20, denoise=False):
    y, sr = librosa.load(audio_path)
    if denoise:
        y = denoise_spectral_subtract(y, sr)
    S_db, time_vals, freq_vals, hop_length = compute_spectrogram(y, sr)

    # threshold per frequency row
    thresh = np.percentile(S_db, percentile, axis=1, keepdims=True)
    S_bin = S_db > thresh

    # find contours in the binary mask
    contours = measure.find_contours(S_bin, 0.5)
    results = []
    for contour in contours:
        x_pix, y_pix = contour[:, 1], contour[:, 0]
        x_time = np.interp(x_pix, np.arange(S_db.shape[1]), time_vals)
        y_freq = np.interp(y_pix, np.arange(S_db.shape[0]), freq_vals)
        if len(x_time) > min_len:
            freq_range = float(np.max(y_freq) - np.min(y_freq))
            if freq_range < float(max_freq_range):
                results.append((float(np.median(y_freq)), x_time, y_freq))

    # sort by median frequency
    results.sort(key=lambda tup: tup[0])

    contours_json = []
    for i, (_, tx, fy) in enumerate(results):
        contours_json.append({
            "name": f"F{i}",
            "time": tx.astype(float).tolist(),
            "freq": fy.astype(float).tolist()
        })

    return {
        "S_db": S_db,
        "sr": sr,
        "hop_length": hop_length,
        "tvals": time_vals,
        "fvals": freq_vals,
        "contours": contours_json
    }

# ---------------------------
# Bark utilities
# ---------------------------
BARK_EDGES = np.array([
    20, 100, 200, 300, 400, 510, 630, 770, 920,
    1080, 1270, 1480, 1720, 2000, 2320, 2700,
    3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
], dtype=float)

def compute_bark_energy(S_db, sr, n_fft):
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    energies = []
    for i in range(len(BARK_EDGES) - 1):
        fmin, fmax = BARK_EDGES[i], BARK_EDGES[i+1]
        band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if band_idx.size == 0:
            energies.append(0.0)
        else:
            energies.append(float(S_db[band_idx, :].mean()))
    return energies

# ---------------------------
# Tables & exports (robust)
# ---------------------------
def _safe_median_positive(arr) -> float:
    """Median of strictly-positive values; returns 0.0 if none."""
    a = np.asarray(arr, dtype=float)
    pos = a[a > 0]
    return float(np.median(pos)) if pos.size else 0.0

def compute_frequency_distance_matrix(contours):
    names = [c["name"] for c in contours]
    means = [np.median(np.array(c["freq"])) for c in contours] if contours else []
    mat = [
        [round(abs(a - b), 2) if i != j else 0 for j, b in enumerate(means)]
        for i, a in enumerate(means)
    ]
    return pd.DataFrame(mat, index=names, columns=names)

def compute_formant_time_ranges(contours):
    rows = []
    for c in contours:
        t = np.array(c["time"])
        rows.append([c["name"], round(t.min(), 2), round(t.max(), 2), round(t.max()-t.min(), 2)])
    return pd.DataFrame(rows, columns=["Formant", "Start Time (s)", "End Time (s)", "Duration (s)"])

def compute_raw_distance_matrix(tracks):
    """
    tracks: list of 1D arrays/lists (per-formant frequency over time).
    Uses median over strictly-positive samples; returns 0.0 if none.
    """
    med = [_safe_median_positive(t) for t in tracks]
    idx = [f"F{i+1}" for i in range(len(med))]
    mat = [
        [round(abs(a - b), 2) if i != j else 0 for j, b in enumerate(med)]
        for i, a in enumerate(med)
    ]
    return pd.DataFrame(mat, index=idx, columns=idx)

def compute_raw_time_ranges(times, tracks):
    """
    Time ranges only for tracks that have any strictly-positive samples.
    """
    data = []
    tarr = np.array(times, dtype=float)
    for i, tr in enumerate(tracks):
        tr = np.asarray(tr, dtype=float)
        nz = np.where(tr > 0)[0]
        if nz.size:
            st, et = tarr[nz[0]], tarr[nz[-1]]
            data.append([f"F{i+1}", round(st, 2), round(et, 2), round(et - st, 2)])
    return pd.DataFrame(data, columns=["Formant", "Start Time (s)", "End Time (s)", "Duration (s)"])
