import plotly.graph_objs as go
import plotly.io as pio
import numpy as np
import librosa

def generate_interactive_raw_spectrogram(S_db, sr, hop_length, times, f0, tracks):
    import plotly.graph_objs as go
    import numpy as np
    import plotly.io as pio
    import librosa

    # Explicit time and frequency axes for the heatmap
    time_vals = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freq_vals = librosa.fft_frequencies(sr=sr, n_fft=(S_db.shape[0] - 1) * 2)

    # Heatmap with real axes
    heatmap = go.Heatmap(
        z=S_db,
        x=time_vals,
        y=freq_vals,
        colorscale='Magma',
        showscale=True,
        colorbar=dict(title='dB'),
    )

    # Overlays for F0 and formants
    overlays = []

    if f0:
        overlays.append(go.Scatter(
            x=times,
            y=f0,
            mode='lines',
            name='F0',
            line=dict(color='cyan', width=2)
        ))

    for i, track in enumerate(tracks):
        if np.any(track):
            overlays.append(go.Scatter(
                x=times,
                y=track,
                mode='lines',
                name=f"F{i+1}",
                line=dict(width=1)
            ))

    layout = go.Layout(
        title="Raw Audio - Interactive Spectrogram with Formants",
        xaxis=dict(title='Time (s)'),
        yaxis=dict(title='Frequency (Hz)'),
        margin=dict(l=50, r=50, t=40, b=50),
        height=600,
        hovermode='closest'
    )

    fig = go.Figure(data=[heatmap] + overlays, layout=layout)
    return pio.to_html(fig, full_html=False)



def generate_interactive_image_spectrogram(S_db, sr, hop_length, contours):
    import plotly.graph_objs as go
    import plotly.io as pio
    import numpy as np
    import librosa

    # Create time and frequency grids
    time_vals = librosa.frames_to_time(np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length)
    freq_vals = librosa.fft_frequencies(sr=sr, n_fft=(S_db.shape[0] - 1) * 2)

    # Plot base spectrogram heatmap
    heatmap = go.Heatmap(
        z=S_db,
        colorscale='Inferno',
        showscale=True,
        colorbar=dict(title='dB'),
        x=time_vals,
        y=freq_vals,
    )

    overlays = []

    for i, (median_freq, x_time_raw, y_freq_raw) in enumerate(contours):
        # Interpolate x_time_raw and y_freq_raw to match the axes
        x_pix = np.interp(x_time_raw, time_vals, np.arange(len(time_vals)))
        y_pix = np.interp(y_freq_raw, freq_vals, np.arange(len(freq_vals)))

        x_time = np.interp(x_pix, np.arange(len(time_vals)), time_vals)
        y_freq = np.interp(y_pix, np.arange(len(freq_vals)), freq_vals)

        if len(x_time) > 0 and len(y_freq) > 0:
            overlays.append(go.Scatter(
                x=x_time,
                y=y_freq,
                mode='lines',
                name=f"F{i}",
                line=dict(width=1)
            ))

    layout = go.Layout(
        title="Image-Based - Interactive Spectrogram with Formants",
        xaxis=dict(title='Time (s)'),
        yaxis=dict(title='Frequency (Hz)'),
        margin=dict(l=50, r=50, t=40, b=50),
        height=600,
        hovermode='closest'
    )

    fig = go.Figure(data=[heatmap] + overlays, layout=layout)
    return pio.to_html(fig, full_html=False)
