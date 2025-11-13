# Spectrogram Tool

An interactive **Flask-based web application** for visualizing, comparing, and analyzing speech audio using both **signal-based** (Parselmouth) and **image-based** (OpenCV + ML) methods.
It provides real-time **spectrograms, formant extraction, denoising, ML-mask cleaning, Bark-band energy plots**, and downloadable CSVs all in a clean web UI.

---

## Features

###  **Formant Analyzer**

* Upload any audio (`.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.aac`)
* Generates an interactive **Plotly spectrogram** with:

  * F0 (pitch contour)
  * Formant tracks (via Parselmouth)
  * Image-based formant contours (OpenCV)
* Apply pre-processing options:

  * `none` - raw input
  * `specsub` - spectral subtraction denoise
  * `mlmask` - machine-learned visual mask (GMM + XGBoost)
* Compute **Bark-band energy distributions**
* Export as CSV:

  * Frequency Distance Matrix
  * Formant Time Ranges
  * Raw Distance Matrix
  * Raw Time Ranges

---

### **Spectrogram Comparator**

* Upload a single file to visualize:

  * **Original spectrogram**
  * **“Cleaned” spectrogram** (ML foreground mask)
* Uses **custom NeonGreen colormap** and OpenCV rendering.
* Built-in audio player for playback.

---

### **Two-File Overlay**

* Upload two audios (Reference + Secondary)
* Choose custom colors for each
* Produces overlaid spectrograms for direct visual comparison

---

## Folder Structure

```
Spectrogram-Tool/
├── app.py                # Flask routes + backend logic
├── audio_utils.py        # DSP utilities, formant extraction, Bark energy
├── requirements.txt      # Base dependencies
├── templates/            # HTML pages (Analyzer, Comparator, Overlay)
│   ├── index.html
│   ├── compare.html
│   └── overlay.html
├── static/
│   ├── app.js            # Analyzer front-end logic
│   ├── compare.js        # Comparator logic
│   ├── style.css         # UI styling (dark theme)
│   └── specs/            # Output spectrogram images
└── .vscode/              # (Optional) editor settings
```

---

## Installation & Setup

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
pip install opencv-python scikit-learn xgboost soundfile
```

> These extras are required at runtime:
>
> * **opencv-python** – for color mapping, contouring, and masking
> * **scikit-learn** – Gaussian Mixture Models
> * **xgboost** – ML classifier for spectrogram mask
> * **soundfile** – for decoding non-WAV audio

If `.m4a` or `.aac` decoding fails, ensure **FFmpeg** is installed.

### 3) Run the app

```bash
python app.py
```

Open your browser and visit  [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## Usage Guide

### Home (`/`)

* Upload your file → choose **Clean Mode**
* Enable overlays: `F0`, `Formants`, or `Contours`
* View interactive Plotly spectrogram
* Download analysis tables (CSV)

### Compare (`/compare`)

* Upload one file → view **Original** vs **Cleaned** spectrograms
* Play audio within the page

### Overlay (`/overlay`)

* Upload two files → select custom colors
* View aligned, colorized overlay visualization

---

##  Key Modules

| File                | Description                                                                                                                                                     |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `app.py`            | Core Flask app with routes for `/analyze`, `/compare`, `/overlay`, `/download/csv`. Handles spectrogram creation, color mapping, ML masking, and CSV streaming. |
| `audio_utils.py`    | DSP core: spectral subtraction, Parselmouth formants, image-based formants, Bark energy, and distance/time matrix computation.                                  |
| `templates/`        | Web UI layout for Analyzer, Comparator, and Overlay pages.                                                                                                      |
| `static/app.js`     | Front-end logic for uploads, toggles, and Plotly rendering.                                                                                                     |
| `static/compare.js` | Comparator UI with live updates and audio player.                                                                                                               |
| `static/specs/`     | Auto-generated PNG spectrograms and overlay outputs.                                                                                                            |

---

##  API Overview

### `/analyze`

Accepts audio + processing params.
**POST (multipart/form-data):**

```
file: <audio>
method: "raw"
params: {"clean_mode": "mlmask"}  # or "none", "specsub"
```

**Returns:** JSON with spectrogram data, contours, Bark energy, and CSVs.

### `/download/csv`

Stream any returned CSV to the browser.

### `/compare/upload`

Generates OG + cleaned PNGs for uploaded audio.

### `/overlay/upload`

Generates two spectrograms with user-selected color overlays.

---

## Technologies Used

* **Flask 3.1** - web backend
* **Librosa 0.10** - audio processing
* **Praat-Parselmouth** - formant extraction
* **OpenCV / scikit-image** - image-based contouring
* **scikit-learn + XGBoost** - spectrogram foreground mask
* **Plotly.js** - dynamic spectrogram visualization
