import librosa
import os
import numpy as np

# Pick a file that exists
test_file = "aufnahmen/conv_H_Waldherr_19-08-2024_11_56_16.mp4"

if not os.path.exists(test_file):
    print(f"File not found: {test_file}")
    files = [f for f in os.listdir("aufnahmen") if f.endswith(".mp4")]
    if files:
        test_file = os.path.join("aufnahmen", files[0])
else:
    print(f"File found: {test_file}")

print(f"Testing librosa.load on: {test_file}")

try:
    y, sr = librosa.load(test_file, sr=None) # sr=None to preserve native if possible, or default 22050
    print(f"Success! Sample rate: {sr}, Shape: {y.shape}, Duration: {len(y)/sr:.2f}s")
except Exception as e:
    print(f"Failed to load with librosa: {e}")
