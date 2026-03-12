from pathlib import Path
import torchaudio
import torch
import gc
from speechbrain.inference.separation import SepformerSeparation as separator
from tqdm import tqdm

from audio_io import normalize_to_wav16k_mono

RAW_DIR = Path("../aufnahmen25")
NORM_DIR = Path("data/normalized")
CHUNKS_DIR = Path("data/chunks")
OUT_CHUNKS_DIR = Path("out_separated_chunks")
OUT_DIR = Path("out_separated")
MODEL_SOURCE = "speechbrain/sepformer-whamr16k"
SAVE_DIR = "pretrained_models/sepformer-whamr16k"

def list_input_files() -> list[Path]:
    files = []
    for p in RAW_DIR.glob("**/*"):
        if p.is_file() and p.suffix.lower() == ".mp4":
            files.append(p)
    return sorted(files)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    NORM_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Sepformer model from {MODEL_SOURCE}...")
    # NOTE: savedir parameter is removed to avoid symlink issues on Windows
    model = separator.from_hparams(source=MODEL_SOURCE)
    print("Model loaded successfully.")

    audio_files = list_input_files()
    if not audio_files:
        print(f"No .mp4 files found in {RAW_DIR}")
        return

    print(f"Found {len(audio_files)} files to process.")

    for src_path in tqdm(audio_files, desc="Separating audio"):
        recording_id = src_path.stem
        norm_path = NORM_DIR / f"{recording_id}.wav"
        
        # 1. Normalize and resample to 16kHz mono
        if not norm_path.exists():
            try:
                normalize_to_wav16k_mono(src_path, norm_path)
            except Exception as e:
                print(f"\nError normalizing {src_path.name}: {e}")
                continue
                
        # 2. Setup output directory for this recording
        rec_out_dir = OUT_DIR / recording_id
        rec_out_dir.mkdir(parents=True, exist_ok=True)
        
        out_spk1 = rec_out_dir / f"{recording_id}_spk1.wav"
        out_spk2 = rec_out_dir / f"{recording_id}_spk2.wav"
        
        # Skip if already processed
        if out_spk1.exists() and out_spk2.exists():
            continue

        # 3. Process the entire normalized file with Sepformer
        try:
            sample_rate = 16000
            
            # Load and separate the full file
            with torch.no_grad():
                est_sources = model.separate_file(path=str(norm_path))
            
            # Save separated tracks
            torchaudio.save(str(out_spk1), est_sources[:, :, 0].detach().cpu(), sample_rate)
            torchaudio.save(str(out_spk2), est_sources[:, :, 1].detach().cpu(), sample_rate)
            
            del est_sources
            gc.collect()
            
        except Exception as e:
            print(f"\nError separating full file {recording_id}: {e}")

if __name__ == "__main__":
    main()
