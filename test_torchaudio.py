try:
    import torchaudio
    import torch
    print("torchaudio available")
    
    test_file = "aufnahmen/conv_H_Waldherr_19-08-2024_11_56_16.mp4"
    if hasattr(torchaudio, 'load'):
        try:
            waveform, sample_rate = torchaudio.load(test_file)
            print(f"Success! SR: {sample_rate}, Shape: {waveform.shape}")
        except Exception as e:
            print(f"torchaudio load failed: {e}")

except ImportError:
    print("torchaudio not found")
