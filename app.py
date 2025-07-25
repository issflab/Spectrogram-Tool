from flask import Flask, request, render_template
from audio_utils import *
import os
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    img_data = dist_matrix = time_df = mode = bark_filter_img = total_formants = None
    dist_html = time_html = ""

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
            total_formants = len(contours)

        elif method == "raw":
            times, f0, tracks, S_db, sr, hop = extract_formants_raw(path)
            img_data = plot_raw_spectrogram_with_tracks(S_db, sr, hop, times, f0, tracks)
            dist_matrix = compute_raw_distance_matrix(tracks)
            time_df = compute_raw_time_ranges(times, tracks)
            mode = "Raw Audio"
            # Count non-empty tracks
            total_formants = sum(1 for track in tracks if np.any(track))

        bark_filter_img = plot_bark_filterbank()
        
        # Convert dataframes to HTML strings
        if dist_matrix is not None:
            dist_html = dist_matrix.to_html()
        if time_df is not None:
            time_html = time_df.to_html()

    return render_template("upload.html",
                           img_data=img_data,
                           dist_html=dist_html,
                           time_html=time_html,
                           mode=mode,
                           bark_filter_img=bark_filter_img,
                           total_formants=total_formants)

if __name__ == "__main__":
    app.run(debug=True)