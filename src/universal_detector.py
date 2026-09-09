import os
import sys
import glob
import warnings
import numpy as np
import librosa
from tensorflow.keras.models import load_model

warnings.filterwarnings("ignore")

# ==========================================
# Exact Parameters Matching Your Model
# ==========================================
MODEL_PATH = "models/cnn_audio_detector.keras"
SAMPLE_RATE = 16000
N_MFCC = 40
MAX_FRAMES = 138       # ~4.41 seconds window
HOP_LENGTH = 512
SLIDE_HOP_SEC = 2.2    # ~50% overlap for long audios
BATCH_SIZE = 32


def load_audio_universal(audio_path, target_sr=SAMPLE_RATE):
    """
    Robust audio loader: supports .wav, .mp3, .m4a, .ogg, .opus, .flac, .aac, .mp4
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"File not found: {audio_path}")

    try:
        signal, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        if len(signal) > 0:
            return signal, sr
    except Exception:
        pass

    # Fallback to pydub for complex containers/codecs
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        samples = np.array(audio.get_array_of_samples())
        signal = samples.astype(np.float32) / (2 ** (audio.sample_width * 8 - 1))
        return signal, target_sr
    except Exception as e:
        raise RuntimeError(f"Could not decode audio '{audio_path}': {e}")


def pad_or_truncate_exact(mfcc, target_frames=MAX_FRAMES):
    """Exact zero-padding matching your trained model."""
    current_frames = mfcc.shape[1]
    if current_frames < target_frames:
        pad_width = target_frames - current_frames
        return np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        return mfcc[:, :target_frames]


def normalize_chunk_exact(chunk):
    """Exact scalar normalization matching your trained model."""
    return (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-8)


def detect_deepfake(audio_path, model):
    """
    Evaluates audio of ANY format and ANY duration.
    """
    # 1. Universal Load
    signal, sr = load_audio_universal(audio_path, target_sr=SAMPLE_RATE)

    # 2. Trim silence
    trimmed_signal, _ = librosa.effects.trim(signal, top_db=25)
    if len(trimmed_signal) >= sr * 0.4:
        signal = trimmed_signal

    duration_sec = len(signal) / sr

    if duration_sec < 0.3:
        return {
            "file": os.path.basename(audio_path),
            "duration": f"{duration_sec:.2f}s",
            "verdict": "ERROR (Audio too short, minimum 0.5s required)",
            "confidence": "0.0%",
            "avg_spoof_prob": "0.0%",
            "max_spoof_prob": "0.0%",
            "fake_segments_count": 0,
            "tampered_intervals": []
        }

    # 3. Extract Full MFCC
    full_mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH
    )
    total_frames = full_mfcc.shape[1]

    chunks = []
    timestamps = []

    # 4. Handle Any Duration (Short vs. Long)
    if total_frames <= MAX_FRAMES:
        # Short audio (< 4.41s)
        chunk = pad_or_truncate_exact(full_mfcc, MAX_FRAMES)
        chunk_norm = normalize_chunk_exact(chunk)
        chunks.append(chunk_norm)
        timestamps.append((0.0, round(duration_sec, 2)))
    else:
        # Long audio: Sliding window with 50% overlap
        slide_hop_frames = int((SLIDE_HOP_SEC * sr) / HOP_LENGTH)
        for start_idx in range(0, total_frames - MAX_FRAMES + 1, slide_hop_frames):
            chunk = full_mfcc[:, start_idx : start_idx + MAX_FRAMES]
            chunk_norm = normalize_chunk_exact(chunk)
            chunks.append(chunk_norm)

            t_start = (start_idx * HOP_LENGTH) / sr
            t_end = ((start_idx + MAX_FRAMES) * HOP_LENGTH) / sr
            timestamps.append((round(t_start, 2), round(min(t_end, duration_sec), 2)))

        # Include tail end if not fully covered
        if (total_frames - MAX_FRAMES) % slide_hop_frames != 0:
            chunk = full_mfcc[:, -MAX_FRAMES:]
            chunk_norm = normalize_chunk_exact(chunk)
            chunks.append(chunk_norm)
            t_start = ((total_frames - MAX_FRAMES) * HOP_LENGTH) / sr
            timestamps.append((round(t_start, 2), round(duration_sec, 2)))

    # 5. Fast Batch Prediction
    X_input = np.array(chunks, dtype=np.float32)
    X_input = np.expand_dims(X_input, axis=-1)  # Shape: (N, 40, 138, 1)

    raw_preds = model.predict(X_input, batch_size=BATCH_SIZE, verbose=0).flatten()
    segment_probs = [float(p) for p in raw_preds]

    # 6. Intelligent Aggregation & Tamper Localization
    # Standard threshold: 0.50. For WhatsApp Opus (.ogg/.opus): 0.85
    is_opus = audio_path.lower().endswith((".ogg", ".opus"))
    spoof_threshold = 0.88 if is_opus else 0.65

    fake_segments = []
    for (t_start, t_end), prob in zip(timestamps, segment_probs):
        if prob >= spoof_threshold:
            fake_segments.append({
                "start": f"{int(t_start // 60):02d}:{t_start % 60:05.2f}",
                "end": f"{int(t_end // 60):02d}:{t_end % 60:05.2f}",
                "spoof_prob": round(prob * 100, 2)
            })

    avg_prob = float(np.mean(segment_probs))
    max_prob = float(np.max(segment_probs))

    # Verdict
    is_spoof = (len(fake_segments) > 0) or (avg_prob >= 0.50)
    verdict = "SPOOF (Fake / Manipulated Voice)" if is_spoof else "BONAFIDE (Real Human Voice)"
    overall_confidence = max_prob if is_spoof else (1.0 - avg_prob)
    
    return {
        "file": os.path.basename(audio_path),
        "duration": f"{duration_sec:.2f}s",
        "verdict": verdict,
        "confidence": f"{overall_confidence * 100:.2f}%",
        "avg_spoof_prob": f"{avg_prob * 100:.2f}%",
        "max_spoof_prob": f"{max_prob * 100:.2f}%",
        "fake_segments_count": len(fake_segments),
        "tampered_intervals": fake_segments
    }


def main():
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python src/universal_detector.py test/sample.mp3")
        print("  python src/universal_detector.py test/\n")
        sys.exit(1)

    target_path = sys.argv[1]

    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at '{MODEL_PATH}'")
        sys.exit(1)

    print(f"Loading Model: {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    print("Model Loaded Successfully!\n")

    if os.path.isdir(target_path):
        exts = ("*.wav", "*.mp3", "*.m4a", "*.ogg", "*.opus", "*.flac", "*.aac", "*.mp4", "*.mkv")
        audio_files = []
        for ext in exts:
            audio_files.extend(glob.glob(os.path.join(target_path, ext)))
        if not audio_files:
            print(f"No audio files found in: {target_path}")
            sys.exit(1)
    else:
        audio_files = [target_path]

    print(f"Found {len(audio_files)} file(s) to inspect.\n" + "=" * 60)

    for fpath in audio_files:
        res = detect_deepfake(fpath, model)

        print(f"Audio File  : {res['file']} (Duration: {res['duration']})")
        print(f"Verdict     : {res['verdict']}")
        print(f"Confidence  : {res['confidence']}")
        print(f"Avg Spoof % : {res['avg_spoof_prob']}  |  Peak Spoof %: {res['max_spoof_prob']}")

        if res["fake_segments_count"] > 0:
            print(f"\n[!] ALERT: {res['fake_segments_count']} Manipulated Segment(s) Detected:")
            for item in res["tampered_intervals"]:
                print(f"    - {item['start']} to {item['end']} --> Spoof Probability: {item['spoof_prob']}%")
        else:
            print("Status      : Clean (No synthetic manipulation detected)")

        print("=" * 60)


if __name__ == "__main__":
    main()