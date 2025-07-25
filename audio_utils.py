import librosa
import librosa.display
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import parselmouth
from skimage import measure
import matplotlib.cm as cm
import pandas as pd
import io
import base64
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib

def extract_formants_raw(audio_path, max_formants=20):
    snd = parselmouth.Sound(audio_path)
    pitch = snd.to_pitch(time_step=0.01)
    formant = snd.to_formant_burg(time_step=0.01)
    y, sr = librosa.load(audio_path, duration=5.0)
    n_fft, hop_length = 2048, 256
    S_db = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)), ref=np.max)

    times = pitch.xs()
    f0_track = []
    formant_tracks = [[] for _ in range(max_formants)]

    for t in times:
        f0 = pitch.get_value_at_time(t)
        f0_track.append(f0 if f0 else 0.0)
        for i in range(1, max_formants + 1):
            try:
                f = formant.get_value_at_time(i, t)
                formant_tracks[i - 1].append(f if f is not None else 0.0)
            except:
                formant_tracks[i - 1].append(0.0)

    return times, f0_track, formant_tracks, S_db, sr, hop_length

def extract_formants_image(audio_path, percentile=90, max_freq_range=500, min_len=20):
    y, sr = librosa.load(audio_path)
    n_fft, hop_length = 2048, 256
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    S_db = np.clip(S_db, -80, 0)

    thresh = np.percentile(S_db, percentile, axis=1, keepdims=True)
    S_bin = S_db > thresh
    contours = measure.find_contours(S_bin, 0.5)

    time_vals = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freq_vals = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    horizontal_contours = []
    for contour in contours:
        x_pix, y_pix = contour[:, 1], contour[:, 0]
        x_time = np.interp(x_pix, np.arange(S_db.shape[1]), time_vals)
        y_freq = np.interp(y_pix, np.arange(S_db.shape[0]), freq_vals)
        if len(x_time) > min_len:
            freq_range = np.max(y_freq) - np.min(y_freq)
            if freq_range < max_freq_range:
                horizontal_contours.append((np.median(y_freq), x_time, y_freq))
    horizontal_contours.sort(key=lambda tup: tup[0])
    return horizontal_contours, S_db, sr, hop_length

def compute_frequency_distance_matrix(contours):
    names = [f"F{i}" for i in range(len(contours))]
    means = [np.median(yf) for (_, _, yf) in contours]
    return pd.DataFrame([[round(abs(a - b), 2) if i != j else 0
                          for j, b in enumerate(means)] for i, a in enumerate(means)],
                        index=names, columns=names)

def compute_formant_time_ranges(contours):
    return pd.DataFrame([(f"F{i}", round(np.min(t), 2), round(np.max(t), 2), round(np.max(t) - np.min(t), 2))
                         for i, (_, t, _) in enumerate(contours)],
                        columns=["Formant", "Start Time (s)", "End Time (s)", "Duration (s)"])

def plot_spectrogram_with_formants(S_db, sr, hop_length, contours):
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='inferno')
    cmap = cm.get_cmap('tab10', len(contours))
    for i, (_, x, y) in enumerate(contours):
        plt.plot(x, y, color=cmap(i), linewidth=1.5, label=f"F{i}")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def plot_raw_spectrogram_with_tracks(S_db, sr, hop_length, times, f0, formants):
    plt.figure(figsize=(14, 6))
    librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', cmap='magma')
    plt.plot(times, f0, color='cyan', label='F0')
    cmap = cm.get_cmap('viridis', len(formants))
    for i, ftrack in enumerate(formants):
        if np.any(ftrack):
            plt.plot(times, ftrack, color=cmap(i), label=f"F{i+1}")
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def compute_raw_distance_matrix(tracks):
    med = [np.median([v for v in t if v > 0]) if np.any(t) else 0 for t in tracks]
    return pd.DataFrame([[round(abs(a - b), 2) if i != j else 0
                          for j, b in enumerate(med)] for i, a in enumerate(med)],
                        index=[f"F{i+1}" for i in range(len(med))],
                        columns=[f"F{i+1}" for i in range(len(med))])

def compute_raw_time_ranges(times, tracks):
    data = []
    for i, t in enumerate(tracks):
        nz = np.where(np.array(t) > 0)[0]
        if nz.size:
            st, et = times[nz[0]], times[nz[-1]]
            data.append((f"F{i+1}", round(st, 2), round(et, 2), round(et - st, 2)))
    return pd.DataFrame(data, columns=["Formant", "Start Time (s)", "End Time (s)", "Duration (s)"])

def plot_bark(S_db, sr, hop_length):
    # Approximate Bark band edges (24 critical bands)
    bark_edges = np.array([
        20, 100, 200, 300, 400, 510, 630, 770, 920,
        1080, 1270, 1480, 1720, 2000, 2320, 2700,
        3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
    ])
    n_fft = (S_db.shape[0] - 1) * 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    bark_energy = []
    for i in range(len(bark_edges) - 1):
        fmin, fmax = bark_edges[i], bark_edges[i+1]
        band_idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        if len(band_idx) == 0:
            bark_energy.append(0)
        else:
            band_power = S_db[band_idx, :].mean()
            bark_energy.append(band_power)

    plt.figure(figsize=(10, 3))
    plt.bar(range(1, len(bark_energy) + 1), bark_energy)
    plt.title("Bark Band Energy")
    plt.xlabel("Bark Band")
    plt.ylabel("Avg dB")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()

def plot_bark_filterbank():
    bark_edges = np.array([
        20, 100, 200, 300, 400, 510, 630, 770, 920,
        1080, 1270, 1480, 1720, 2000, 2320, 2700,
        3150, 3700, 4400, 5300, 6400, 7700, 9500, 12000, 15500
    ])
    plt.figure(figsize=(8, 5))
    for i in range(len(bark_edges) - 1):
        left = bark_edges[i]
        right = bark_edges[i+1]
        center = (left + right) / 2
        height = center / 100
        plt.bar(i + 1, height, width=1, align='center', color='navy')
    plt.yscale('log')
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Band")
    plt.title("Bark Scale Filter Bank (Approximate)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode()