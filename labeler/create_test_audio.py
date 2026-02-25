import numpy as np
import soundfile as sf

def create_test_audio(filename="test_audio.wav", duration=3.0, fs=24000):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Generate a 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    print(f"Generating {filename}...")
    sf.write(filename, audio, fs)
    print("Done.")

if __name__ == "__main__":
    create_test_audio()
