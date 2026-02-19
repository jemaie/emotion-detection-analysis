import soundfile as sf
import os

# Pick a file that exists
test_file = "aufnahmen/conv_H_Waldherr_19-08-2024_11_56_16.mp4"

if not os.path.exists(test_file):
    print(f"File not found: {test_file}")
    # Try to find any mp4
    files = [f for f in os.listdir("aufnahmen") if f.endswith(".mp4")]
    if files:
        test_file = os.path.join("aufnahmen", files[0])
        print(f"Using alternative file: {test_file}")
    else:
        print("No MP4 files found.")
        exit(1)

print(f"Testing read on: {test_file}")

try:
    data, fs = sf.read(test_file)
    print(f"Success! Sample rate: {fs}, Shape: {data.shape}")
except Exception as e:
    print(f"Failed to read MP4 with soundfile: {e}")
