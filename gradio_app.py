import os
import warnings
import numpy as np
import librosa
import matplotlib.pyplot as plt
import gradio as gr
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# ==========================================
# Settings & Model Loading
# ==========================================
MODEL_PATH = "models/cnn_audio_detector.keras"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 138       # ~4.41s window
HOP_LENGTH = 512
SLIDE_HOP_SEC = 2.2

print(f"Loading Model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)
print("Model Loaded Successfully!")


def pad_or_truncate_exact(mfcc, target_frames=MAX_FRAMES):
    current_frames = mfcc.shape[1]
    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    return mfcc[:, :target_frames]


def normalize_chunk_exact(chunk):
    return (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-8)


def plot_audio_visuals(signal, sr, mfcc):
    """Generates high-contrast Waveform and 40-Band MFCC Spectrogram."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.6), gridspec_kw={'height_ratios': [1, 1.25]})
    fig.patch.set_facecolor("#0f172a")

    # 1. Waveform Plot
    times = np.linspace(0, len(signal) / sr, len(signal))
    axes[0].plot(times, signal, color="#38bdf8", lw=1.2)
    axes[0].set_title("Audio Waveform Dynamics", color="#f8fafc", fontsize=11, fontweight="bold", pad=6)
    axes[0].set_ylabel("Amplitude", color="#94a3b8", fontsize=9)
    axes[0].tick_params(colors="#94a3b8", labelsize=8)
    axes[0].set_facecolor("#1e293b")
    axes[0].grid(True, color="#334155", linestyle="--", alpha=0.6)

    # 2. MFCC Heatmap
    img = axes[1].imshow(
        mfcc,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, len(signal) / sr, 0, N_MFCC]
    )
    axes[1].set_title("40-Band MFCC Spectrogram (Acoustic Feature Map)", color="#f8fafc", fontsize=11, fontweight="bold", pad=6)
    axes[1].set_xlabel("Timeline (Seconds)", color="#94a3b8", fontsize=9)
    axes[1].set_ylabel("MFCC Index", color="#94a3b8", fontsize=9)
    axes[1].tick_params(colors="#94a3b8", labelsize=8)
    axes[1].set_facecolor("#1e293b")

    cbar = fig.colorbar(img, ax=axes[1], pad=0.02)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=8)
    cbar.set_label("Energy (dB)", color="#94a3b8", fontsize=8)

    plt.tight_layout()
    return fig


def analyze_audio_ui(audio_path):
    """Main processing function called by the Gradio Web Interface."""
    if audio_path is None:
        return "⚠️ Please upload an audio file or record from microphone.", "", None, None, None

    # 1. Load Audio
    try:
        signal, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
    except Exception:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
        samples = np.array(audio.get_array_of_samples())
        signal = samples.astype(np.float32) / (2 ** (audio.sample_width * 8 - 1))
        sr = SAMPLE_RATE

    duration = len(signal) / sr
    if duration < 0.3:
        return "❌ Audio too short. Please provide at least 0.5s of speech.", "", None, None, None

    # 2. Extract Full MFCC
    full_mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=N_MFCC, hop_length=HOP_LENGTH)
    total_frames = full_mfcc.shape[1]

    # 3. Generate Visual Plot
    spectrogram_plot = plot_audio_visuals(signal, sr, full_mfcc)

    # 4. Chunking & Sliding Window
    chunks = []
    timestamps = []

    if total_frames <= MAX_FRAMES:
        chunk = pad_or_truncate_exact(full_mfcc, MAX_FRAMES)
        chunks.append(normalize_chunk_exact(chunk))
        timestamps.append((0.0, round(duration, 2)))
    else:
        slide_hop_frames = int((SLIDE_HOP_SEC * sr) / HOP_LENGTH)
        for start_idx in range(0, total_frames - MAX_FRAMES + 1, slide_hop_frames):
            chunk = full_mfcc[:, start_idx : start_idx + MAX_FRAMES]
            chunks.append(normalize_chunk_exact(chunk))
            t_start = (start_idx * HOP_LENGTH) / sr
            t_end = ((start_idx + MAX_FRAMES) * HOP_LENGTH) / sr
            timestamps.append((round(t_start, 2), round(min(t_end, duration), 2)))

        if (total_frames - MAX_FRAMES) % slide_hop_frames != 0:
            chunk = full_mfcc[:, -MAX_FRAMES:]
            chunks.append(normalize_chunk_exact(chunk))
            t_start = ((total_frames - MAX_FRAMES) * HOP_LENGTH) / sr
            timestamps.append((round(t_start, 2), round(duration, 2)))

    # 5. Model Inference
    X_input = np.array(chunks, dtype=np.float32)[..., np.newaxis]
    raw_preds = model.predict(X_input, batch_size=32, verbose=0).flatten()
    segment_probs = [float(p) for p in raw_preds]

    avg_spoof = float(np.mean(segment_probs))
    max_spoof = float(np.max(segment_probs))

    # 6. Thresholds (Calibrated for real-world speech & WhatsApp)
    is_opus = audio_path.lower().endswith((".ogg", ".opus"))

    if is_opus:
        # WhatsApp/Telegram Opus audio has a higher baseline due to codec compression
        spoof_threshold = 0.95
        avg_threshold = 0.82
    else:
        # Standard uncompressed audio (.wav, .m4a, high-quality .mp3)
        spoof_threshold = 0.65
        avg_threshold = 0.50

    is_spoof = (max_spoof >= spoof_threshold) or (avg_spoof >= avg_threshold)
    overall_confidence = max_spoof if is_spoof else (1.0 - avg_spoof)

    # 7. Styled Verdict Badge HTML
    if is_spoof:
        verdict_badge = f"""
        <div style="background: linear-gradient(135deg, rgba(239, 68, 68, 0.22) 0%, rgba(185, 28, 28, 0.12) 100%);
                    border: 2px solid #ef4444; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(239, 68, 68, 0.25);">
            <h2 style="color: #f87171; margin: 0; font-size: 24px; font-weight: 800; letter-spacing: 0.5px;">
                ⚠️ SPOOF DETECTED (SYNTHETIC / AI VOICE)
            </h2>
            <p style="color: #e2e8f0; margin: 6px 0 0; font-size: 15px;">
                Acoustic manipulation artifacts detected. Peak Spoof Confidence: <b style="color: #fca5a5; font-size: 17px;">{overall_confidence * 100:.1f}%</b>
            </p>
        </div>
        """
    else:
        verdict_badge = f"""
        <div style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.22) 0%, rgba(21, 128, 61, 0.12) 100%);
                    border: 2px solid #22c55e; border-radius: 12px; padding: 20px; text-align: center; box-shadow: 0 4px 15px rgba(34, 197, 94, 0.25);">
            <h2 style="color: #4ade80; margin: 0; font-size: 24px; font-weight: 800; letter-spacing: 0.5px;">
                ✅ BONAFIDE (AUTHENTIC HUMAN VOICE)
            </h2>
            <p style="color: #e2e8f0; margin: 6px 0 0; font-size: 15px;">
                Natural vocal dynamics verified. Authenticity Confidence: <b style="color: #86efac; font-size: 17px;">{overall_confidence * 100:.1f}%</b>
            </p>
        </div>
        """

    # 8. KPI Stat Cards HTML
    stats_card = f"""
    <div style="display: flex; gap: 10px; margin-top: 14px;">
        <div style="flex: 1; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: 600;">Duration</div>
            <div style="color: #ffffff; font-size: 19px; font-weight: 700; margin-top: 3px;">{duration:.2f}s</div>
        </div>
        <div style="flex: 1; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: 600;">Windows</div>
            <div style="color: #ffffff; font-size: 19px; font-weight: 700; margin-top: 3px;">{len(segment_probs)}</div>
        </div>
        <div style="flex: 1; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: 600;">Peak Spoof</div>
            <div style="color: {'#f87171' if max_spoof >= spoof_threshold else '#4ade80'}; font-size: 19px; font-weight: 700; margin-top: 3px;">{max_spoof * 100:.1f}%</div>
        </div>
        <div style="flex: 1; background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px; text-align: center;">
            <div style="color: #94a3b8; font-size: 11px; text-transform: uppercase; font-weight: 600;">Avg Spoof</div>
            <div style="color: {'#f87171' if avg_spoof >= avg_threshold else '#4ade80'}; font-size: 19px; font-weight: 700; margin-top: 3px;">{avg_spoof * 100:.1f}%</div>
        </div>
    </div>
    """

    # 9. Probability Breakdown
    scores_dict = {
        "Authentic Human Voice": round(1.0 - avg_spoof, 4),
        "Synthetic / Spoof Voice": round(avg_spoof, 4)
    }

    # 10. Segment Timeline Table
    timeline_data = []
    for (t_start, t_end), prob in zip(timestamps, segment_probs):
        status = "🔴 FAKE" if prob >= spoof_threshold else "🟢 REAL"
        timeline_data.append([f"{t_start:.2f}s - {t_end:.2f}s", f"{prob * 100:.1f}%", status])

    return verdict_badge, stats_card, scores_dict, timeline_data, spectrogram_plot


# ==========================================
# Gradio UI Layout with Custom Styling
# ==========================================
custom_css = """
body {
    background-color: #0b0f19;
}
#hero-banner {
    background: linear-gradient(135deg, #111827 0%, #1e293b 100%);
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.badge {
    background-color: #0f172a;
    border: 1px solid #38bdf8;
    color: #38bdf8;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 12px;
    font-weight: 600;
    margin-right: 6px;
    display: inline-block;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"), css=custom_css, title="Deepfake Audio Forensic Engine") as demo:
    gr.HTML("""
    <div id="hero-banner">
        <h1 style="color: #ffffff; margin: 0; font-size: 27px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
            🛡️An Intelligent Speech Authentication System Using MFCC and CNN.
        </h1>
        <p style="color: #94a3b8; margin: 6px 0 12px; font-size: 15px;">
            Acoustic Forensics & Synthetic Speech Localization using 40-Band MFCC and 2D CNN.
        </p>
        <div>
            <span class="badge">ASVspoof 2019 Benchmark</span>
            <span class="badge">TensorFlow 2D CNN</span>
            <span class="badge">40-Band MFCC Spectrogram</span>
            <span class="badge">Any Format & Any Duration</span>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="📁 Upload Audio or Record Live Voice"
            )
            submit_btn = gr.Button("🔍 Run Forensic Analysis", variant="primary", size="lg")
            
            gr.Markdown("""
            ---
            **System Features:**
            * Supports: `.wav`, `.mp3`, `.m4a`, `.ogg`, `.flac`, `.aac`
            * Evaluates arbitrary durations (0.5s to 30+ minutes).
            * Calibrated for standard voice recordings & WhatsApp Opus audio.
            """)

        with gr.Column(scale=2):
            verdict_output = gr.HTML(label="Verdict Banner")
            stats_output = gr.HTML(label="Forensic Metrics")
            prob_label = gr.Label(label="Classification Probability Distribution", num_top_classes=2)
            
            with gr.Tabs():
                with gr.TabItem("📊 Acoustic Waveform & Spectrogram"):
                    plot_output = gr.Plot(label="Acoustic Spectrogram")

                with gr.TabItem("⏱️ Tamper Timeline Breakdown"):
                    timeline_table = gr.Dataframe(
                        headers=["Time Window", "Spoof Probability", "Verdict"],
                        datatype=["str", "str", "str"],
                        label="Tampered Intervals"
                    )

    submit_btn.click(
        fn=analyze_audio_ui,
        inputs=audio_input,
        outputs=[verdict_output, stats_output, prob_label, timeline_table, plot_output]
    )

if __name__ == "__main__":
    # Launch locally (share=False prevents socket 10054 connection crashes)
    demo.launch(share=True,
    inbrowser=True,
    show_error=True)
    # demo.launch(server_name="0.0.0.0", server_port=7860)