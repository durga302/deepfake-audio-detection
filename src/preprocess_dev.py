import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

# ===============================
# Load Labels
# ===============================
labels = pd.read_csv("dataset/dev_labels.csv")

print(labels.head())
print("\nTotal Files:", len(labels))

# ===============================
# Paths
# ===============================
AUDIO_FOLDER = "dataset/LA/ASVspoof2019_LA_dev/flac"

OUTPUT_FOLDER = "processed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===============================
# Parameters
# ===============================
N_MFCC = 40
MAX_FRAMES = 138

X = []
y = []

# ===============================
# Label Encoding
# ===============================
def label_to_number(label):
    if label == "bonafide":
        return 0
    return 1

# ===============================
# Pad / Truncate Function
# ===============================
def pad_or_truncate(mfcc):

    current_frames = mfcc.shape[1]

    if current_frames < MAX_FRAMES:

        pad_width = MAX_FRAMES - current_frames

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :MAX_FRAMES]

    return mfcc


# ===============================
# Feature Extraction
# ===============================
for index, row in tqdm(labels.iterrows(), total=len(labels)):

    audio_path = os.path.join(
        AUDIO_FOLDER,
        row["file_name"]
    )

    try:

        signal, sr = librosa.load(audio_path, sr=16000)

        mfcc = librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=N_MFCC
        )

        # Make every MFCC same size
        mfcc = pad_or_truncate(mfcc)

        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

        X.append(mfcc)

        y.append(label_to_number(row["label"]))

    except Exception as e:

        print(f"\nError processing {row['file_name']}")
        print(e)


# ===============================
# Summary
# ===============================
print("\nProcessed Files :", len(X))

print("\nLabel Distribution:")
print(Counter(y))

print("\nChecking MFCC Shapes...")

shape_counter = Counter([mfcc.shape for mfcc in X])

print(shape_counter)

# ===============================
# Convert to NumPy
# ===============================
X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# Add CNN channel dimension
X = np.expand_dims(X, axis=-1)

print("\nFinal Shapes")
print("X :", X.shape)
print("y :", y.shape)

# ===============================
# Save
# ===============================
np.save("processed/X_dev.npy", X)
np.save("processed/y_dev.npy", y)

print("\nDevelopment dataset saved successfully!")