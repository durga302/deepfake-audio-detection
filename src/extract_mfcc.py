import librosa
import librosa.display
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------------------
# Read the first audio filename
# ------------------------------------
labels = pd.read_csv("dataset/train_labels.csv")

audio_file = labels.iloc[0]["file_name"]

audio_path = f"dataset/LA/ASVspoof2019_LA_train/flac/{audio_file}"

print("Audio File:", audio_file)
print("Path:", audio_path)

# ------------------------------------
# Load Audio
# ------------------------------------
signal, sr = librosa.load(audio_path, sr=16000)

print("\nSample Rate:", sr)
print("Signal Length:", len(signal))

# ------------------------------------
# Extract MFCC
# ------------------------------------
mfcc = librosa.feature.mfcc(
    y=signal,
    sr=sr,
    n_mfcc=40
)

print("\nMFCC Shape:", mfcc.shape)

# ------------------------------------
# Display MFCC
# ------------------------------------
plt.figure(figsize=(12,5))

librosa.display.specshow(
    mfcc,
    sr=sr,
    x_axis="time"
)

plt.colorbar()
plt.title("MFCC Feature Map")

plt.tight_layout()
plt.show()