from flask import Flask, request, render_template
from audio_utils import *
import os

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    img_data = dist_matrix = time_df = mode = bark_filter_img = None

    if request.method == "POST":
        file = request.files["audio"]
        method = request.form["method"]
        path = os.path.join(UPLOAD_DIR, file.filename)
        file.save(path)

        if method == "image":
            contours, S_db, sr, hop = extract_formants_image(path)
            img_data = plot_spectrogram_with_formants(S_db, sr, hop, contours)
            dist_matrix = compute_frequency_distance_matrix(contours)
            time_df = compute_formant_time_ranges(contours)
            mode = "Image-Based"

        elif method == "raw":
            times, f0, tracks, S_db, sr, hop = extract_formants_raw(path)
            img_data = plot_raw_spectrogram_with_tracks(S_db, sr, hop, times, f0, tracks)
            dist_matrix = compute_raw_distance_matrix(tracks)
            time_df = compute_raw_time_ranges(times, tracks)
            mode = "Raw Audio"

        bark_filter_img = plot_bark_filterbank()

    return render_template("upload.html",
                           img_data=img_data,
                           dist=dist_matrix.to_html() if dist_matrix is not None else None,
                           time=time_df.to_html() if time_df is not None else None,
                           mode=mode,
                           bark_filter_img=bark_filter_img)

if __name__ == "__main__":
    app.run(debug=True)